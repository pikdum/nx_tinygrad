defmodule NxTinygrad.Worker do
  @moduledoc """
  Owns a single Python worker OS process behind an Erlang Port and serializes
  requests to it.

  The Port is opened with `packet: 4`, so Erlang reassembles each response frame
  and delivers it as one `{port, {:data, payload}}` message; no manual buffering
  is needed. Requests and responses are correlated by a monotonic request id.

  A worker crash (Port exit) terminates this GenServer, and the supervisor
  restarts it with a fresh generation. In-flight callers are told the worker
  crashed rather than being left to time out.
  """
  use GenServer
  require Logger

  alias NxTinygrad.{Config, Device, Generation, Protocol}

  @registry NxTinygrad.WorkerRegistry
  @protocol_version 1
  @handshake_timeout 30_000

  # -- public API ---------------------------------------------------------

  def start_link(opts) do
    name = Keyword.get(opts, :name, :default)
    GenServer.start_link(__MODULE__, opts, name: via(name))
  end

  @doc "The `:via` tuple used to register/reach a worker by logical name."
  def via(name), do: {:via, Registry, {@registry, {:worker, name}}}

  @doc "Look up a worker pid by name."
  def whereis(pid) when is_pid(pid) do
    if Process.alive?(pid), do: pid
  end

  def whereis(name) do
    case Registry.lookup(@registry, {:worker, name}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  end

  @doc """
  Send a request and await the response.

  Returns `{:ok, result_map, blobs}` on success, or `{:error, exception}` on a
  structured worker error, timeout, or crash.
  """
  @spec request(term() | pid(), String.t(), map(), [binary()], keyword()) ::
          {:ok, map(), [binary()]} | {:error, Exception.t()}
  def request(worker, command, args \\ %{}, blobs \\ [], opts \\ []) do
    timeout = Keyword.get(opts, :timeout, Config.execute_timeout())
    server = resolve(worker)
    # Give the GenServer.call a little longer than the internal timer so the
    # worker's own timeout reply wins over a caller-side exit.
    GenServer.call(server, {:request, command, args, blobs, timeout}, timeout + 5_000)
  end

  @doc "Like `request/5` but raises on error and returns `{result_map, blobs}`."
  def request!(worker, command, args \\ %{}, blobs \\ [], opts \\ []) do
    case request(worker, command, args, blobs, opts) do
      {:ok, result, blobs} -> {result, blobs}
      {:error, exception} -> raise exception
    end
  end

  @doc "Handshake/device info captured at startup."
  def info(worker), do: GenServer.call(resolve(worker), :info)

  @doc "The worker's current generation."
  def generation(worker), do: GenServer.call(resolve(worker), :generation)

  defp resolve(pid) when is_pid(pid), do: pid
  defp resolve(name), do: via(name)

  # -- GenServer ----------------------------------------------------------

  @impl true
  def init(opts) do
    Process.flag(:trap_exit, true)
    name = Keyword.get(opts, :name, :default)
    device_spec = Keyword.get(opts, :device, Config.device())
    generation = Generation.next()

    port = open_port(device_spec, generation)

    case handshake(port) do
      {:ok, hello} ->
        Logger.debug(
          "nx_tinygrad worker #{inspect(name)} up: generation #{generation}, device #{device_spec}"
        )

        state = %{
          name: name,
          device_spec: device_spec,
          port: port,
          generation: generation,
          hello: hello,
          next_req_id: 1,
          pending: %{}
        }

        {:ok, state}

      {:error, reason} ->
        safe_close(port)
        {:stop, reason}
    end
  end

  @impl true
  def handle_call({:request, command, args, blobs, timeout}, from, state) do
    req_id = state.next_req_id
    frame = Protocol.encode(req_id, %{"command" => command, "args" => args}, blobs)
    Port.command(state.port, frame)

    timer = Process.send_after(self(), {:request_timeout, req_id}, timeout)
    pending = Map.put(state.pending, req_id, {from, timer, command})
    {:noreply, %{state | next_req_id: req_id + 1, pending: pending}}
  end

  def handle_call(:info, _from, state), do: {:reply, state.hello, state}
  def handle_call(:generation, _from, state), do: {:reply, state.generation, state}

  @impl true
  def handle_info({port, {:data, payload}}, %{port: port} = state) do
    case Protocol.decode(payload) do
      {:ok, {req_id, meta, blobs}} ->
        {:noreply, deliver(state, req_id, meta, blobs)}

      {:error, reason} ->
        Logger.error(
          "nx_tinygrad worker #{inspect(state.name)} sent an undecodable frame: #{inspect(reason)}"
        )

        {:stop, {:protocol_error, reason}, fail_all(state, protocol_error(reason))}
    end
  end

  def handle_info({:request_timeout, req_id}, state) do
    case Map.pop(state.pending, req_id) do
      {nil, _} ->
        {:noreply, state}

      {{from, _timer, _command}, pending} ->
        GenServer.reply(from, {:error, timeout_error(state)})
        {:noreply, %{state | pending: pending}}
    end
  end

  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.error("nx_tinygrad worker #{inspect(state.name)} exited (status #{status})")

    :telemetry.execute([:nx_tinygrad, :worker, :restart], %{}, %{
      name: state.name,
      generation: state.generation,
      exit_status: status
    })

    {:stop, {:worker_exit, status}, fail_all(state, crash_error(status, state))}
  end

  def handle_info({:EXIT, port, reason}, %{port: port} = state) do
    {:stop, {:port_exit, reason}, fail_all(state, crash_error(reason, state))}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(_reason, state) do
    safe_close(state.port)
    :ok
  end

  # -- internals ----------------------------------------------------------

  defp open_port(device_spec, generation) do
    parsed = Device.parse(device_spec)

    env =
      %{
        "DEV" => parsed.dev,
        "NX_TINYGRAD_DEVICE" => parsed.spec,
        "NX_TINYGRAD_GENERATION" => Integer.to_string(generation),
        "DEBUG" => Integer.to_string(Config.debug()),
        "PYTHONUNBUFFERED" => "1"
      }
      |> Map.merge(parsed.env)
      |> Enum.map(fn {k, v} -> {String.to_charlist(k), String.to_charlist(v)} end)

    Port.open(
      {:spawn_executable, Config.python_executable()},
      [
        :binary,
        :exit_status,
        {:packet, 4},
        {:args, [Config.worker_main()]},
        {:env, env}
      ]
    )
  end

  # Synchronous handshake before entering the message loop.
  defp handshake(port) do
    frame =
      Protocol.encode(0, %{"command" => "hello", "args" => %{"protocol_version" => @protocol_version}}, [])

    Port.command(port, frame)

    receive do
      {^port, {:data, payload}} ->
        case Protocol.decode(payload) do
          {:ok, {_id, %{"ok" => true, "result" => hello}, _}} ->
            verify_protocol(hello)

          {:ok, {_id, %{"ok" => false, "error" => error}, _}} ->
            {:error, {:handshake_rejected, error}}

          {:error, reason} ->
            {:error, {:handshake_decode, reason}}
        end

      {^port, {:exit_status, status}} ->
        {:error, {:worker_exit_during_handshake, status}}
    after
      @handshake_timeout ->
        {:error, :handshake_timeout}
    end
  end

  defp verify_protocol(%{"protocol_version" => v} = hello) when v == @protocol_version, do: {:ok, hello}
  defp verify_protocol(%{"protocol_version" => v}), do: {:error, {:protocol_mismatch, v, @protocol_version}}
  defp verify_protocol(_), do: {:error, :handshake_missing_version}

  defp deliver(state, req_id, meta, blobs) do
    case Map.pop(state.pending, req_id) do
      {nil, _} ->
        Logger.warning("nx_tinygrad worker #{inspect(state.name)} got a reply for unknown req #{req_id}")
        state

      {{from, timer, command}, pending} ->
        Process.cancel_timer(timer)
        GenServer.reply(from, response(meta, blobs, state, command))
        %{state | pending: pending}
    end
  end

  defp fail_all(state, error) do
    for {_id, {from, timer, _command}} <- state.pending do
      Process.cancel_timer(timer)
      GenServer.reply(from, {:error, error})
    end

    %{state | pending: %{}}
  end

  # Translate a decoded response frame into a caller reply.
  defp response(%{"ok" => true, "result" => result}, blobs, _state, _command),
    do: {:ok, result, blobs}

  defp response(%{"ok" => false, "error" => error}, _blobs, state, command) do
    {:error, worker_error(error, state, command)}
  end

  defp response(_meta, _blobs, _state, _command),
    do: {:error, %NxTinygrad.ProtocolError{message: "malformed response frame"}}

  defp worker_error(error, state, command) do
    %NxTinygrad.WorkerError{
      message: Map.get(error, "message", "worker error"),
      class: Map.get(error, "class", "WorkerError"),
      command: command,
      generation: state.generation,
      device: state.device_spec,
      details: Map.get(error, "details", %{}),
      python_traceback: Map.get(error, "python_traceback")
    }
  end

  defp timeout_error(state) do
    %NxTinygrad.WorkerError{
      message: "request timed out",
      class: "Timeout",
      generation: state.generation,
      device: state.device_spec,
      details: %{}
    }
  end

  defp crash_error(status, state) do
    %NxTinygrad.WorkerCrashedError{exit_status: status, generation: state.generation}
  end

  defp protocol_error(reason) do
    %NxTinygrad.ProtocolError{message: "undecodable worker frame: #{inspect(reason)}"}
  end

  defp safe_close(port) do
    if is_port(port) and Port.info(port) != nil do
      Port.close(port)
    end
  rescue
    _ -> :ok
  end
end
