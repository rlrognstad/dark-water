from dark_water.depletion_watchlist.ingest import stations


def test_download_ggmn_stations_requests_global_groundwater_only(monkeypatch):
    captured = {}

    def fake_get_stations(*, network=None, compartment=None, basin=None):
        captured["network"] = network
        captured["compartment"] = compartment
        captured["basin"] = basin
        return "sentinel-result"

    monkeypatch.setattr(stations, "get_stations", fake_get_stations)

    result = stations.download_ggmn_stations()

    assert result == "sentinel-result"
    assert captured["network"] == "ggmn"
    assert captured["compartment"] == "GW"
    assert captured["basin"] is None
