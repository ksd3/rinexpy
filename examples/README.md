# Examples

Each script is self-contained and uses files from the bundled
`tests/data/` corpus, so you can run them straight after
`uv sync`:

```sh
uv run python examples/01_basic_load_and_plot.py
```

| # | Script | What it shows |
|---|--------|---------------|
| 01 | `01_basic_load_and_plot.py` | `load()` + `plots.timeseries()` |
| 02 | `02_batch_convert.py` | parallel `batch_convert()` to NetCDF |
| 03 | `03_streaming_huge_file.py` | `iter_obs3_epochs` for files larger than RAM |
| 04 | `04_sp3_interpolation.py` | `load_sp3` + `interpolate_sp3` to an arbitrary epoch |
| 05 | `05_skyplot_from_nav.py` | NAV to Keplerian to ECEF to az/el to polar plot |
| 06 | `06_spp_positioning.py` | single-point positioning from synthetic OBS+NAV |
| 07 | `07_rtk_baseline.py` | `rtk_fix` with LAMBDA on synthetic dual-receiver data |
| 08 | `08_ntrip_to_rtcm.py` | NTRIP byte stream to `rtcm3.iter_messages` (offline replay) |

Most scripts print numeric output. The plot scripts (`01`, `05`)
show a matplotlib window unless you pass `--save out.png`.

To use your own files, edit the `MAIN_FILE` constant at the top of
each script to point at a RINEX, NetCDF, or SP3 file.
