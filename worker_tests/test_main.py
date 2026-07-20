import device
from main import Handler


def test_synchronize_failure_is_returned_to_the_client():
    handler = Handler(device.parse_device("CPU"))

    class BrokenDevices:
        def __getitem__(self, _name):
            raise RuntimeError("device synchronization failed")

    handler.Device = BrokenDevices()
    meta, blobs = handler.handle({"command": "synchronize", "args": {}}, [])

    assert meta["ok"] is False
    assert meta["error"]["class"] == "RuntimeError"
    assert "synchronization failed" in meta["error"]["message"]
    assert blobs == []
