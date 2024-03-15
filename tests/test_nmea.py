"""Tests for the NMEA-0183 decoder."""

from __future__ import annotations

from datetime import date, datetime, time

from rinexpy.nmea import checksum, iter_lines, parse_sentence

# Real NMEA samples from the GPSd test corpus.
GGA_SAMPLE = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
RMC_SAMPLE = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
GSA_SAMPLE = "$GPGSA,A,3,04,05,,09,12,,,24,,,,,2.5,1.3,2.1*39"
GSV_SAMPLE = "$GPGSV,2,1,08,01,40,083,46,02,17,308,41,12,07,344,39,14,22,228,45*75"
VTG_SAMPLE = "$GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48"


def test_checksum_known_value():
    # Known correct checksum for the canonical GGA sample.
    body = GGA_SAMPLE[1:].split("*", 1)[0]
    assert checksum(body) == 0x47


def test_checksum_accepts_full_sentence():
    assert checksum(GGA_SAMPLE) == 0x47


def test_parse_gga():
    msg = parse_sentence(GGA_SAMPLE)
    assert msg is not None
    assert msg["talker"] == "GP"
    assert msg["type"] == "GGA"
    assert msg["lat"] == 48 + 7.038 / 60
    assert msg["lon"] == 11 + 31.0 / 60
    assert msg["fix_quality"] == 1
    assert msg["n_sat"] == 8
    assert msg["altitude_m"] == 545.4


def test_parse_rmc_combines_date_and_time():
    msg = parse_sentence(RMC_SAMPLE)
    assert msg is not None
    assert msg["status"] == "A"
    assert msg["date"] == date(1994, 3, 23)
    assert msg["time"] == time(12, 35, 19)
    assert msg["datetime"] == datetime(1994, 3, 23, 12, 35, 19)


def test_parse_gsa_dops():
    msg = parse_sentence(GSA_SAMPLE)
    assert msg is not None
    assert msg["fix_type"] == 3
    assert msg["pdop"] == 2.5
    assert msg["hdop"] == 1.3
    assert msg["vdop"] == 2.1
    assert msg["sv_ids"] == [4, 5, 9, 12, 24]


def test_parse_gsv_satellites():
    msg = parse_sentence(GSV_SAMPLE)
    assert msg is not None
    assert msg["sv_in_view"] == 8
    assert msg["satellites"][0] == {
        "prn": 1, "elevation_deg": 40, "azimuth_deg": 83, "snr_dbhz": 46,
    }


def test_parse_vtg():
    msg = parse_sentence(VTG_SAMPLE)
    assert msg is not None
    assert msg["course_true_deg"] == 54.7
    assert msg["speed_kn"] == 5.5
    assert msg["speed_kmh"] == 10.2


def test_bad_checksum_returns_none():
    bad = GGA_SAMPLE[:-2] + "00"  # corrupt the checksum
    assert parse_sentence(bad) is None


def test_unknown_talker_still_decodes():
    """Other talker prefixes (e.g. GN for multi-GNSS) decode the same."""
    line = GGA_SAMPLE.replace("$GP", "$GN")
    # Recompute checksum after substitution.
    body = line[1:].split("*", 1)[0]
    cs = checksum(body)
    line = line[:-2] + f"{cs:02X}"
    msg = parse_sentence(line)
    assert msg is not None
    assert msg["talker"] == "GN"
    assert msg["type"] == "GGA"


def test_iter_lines_skips_garbage():
    lines = ["garbage", GGA_SAMPLE, "", RMC_SAMPLE, "$BADSENT,no,checksum"]
    msgs = list(iter_lines(lines))
    assert len(msgs) == 2
    assert msgs[0]["type"] == "GGA"
    assert msgs[1]["type"] == "RMC"


def test_unknown_sentence_type_returns_raw():
    body = "GPZZZ,foo,bar"
    line = f"$GPZZZ,foo,bar*{checksum(body):02X}"
    msg = parse_sentence(line)
    assert msg is not None
    assert msg["type"] == "ZZZ"
    assert msg["fields"] == ["foo", "bar"]
