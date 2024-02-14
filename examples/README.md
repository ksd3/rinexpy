# Examples

Each script is self-contained and uses files from the bundled
`tests/data/` corpus, so you can run them straight after `pip install -e .`:

```sh
python examples/01_basic_load_and_plot.py
```

| # | Script | What it shows |
|---|--------|---------------|
| 01 | `01_basic_load_and_plot.py` | `load()` + `plots.timeseries()` end-to-end |
| 02 | `02_batch_convert.py` | parallel `batch_convert()` to NetCDF |
| 03 | `03_streaming_huge_file.py` | `iter_obs3_epochs` for files larger than RAM |
| 04 | `04_sp3_interpolation.py` | `load_sp3` + `interpolate_sp3` to arbitrary epoch |
| 05 | `05_skyplot_from_nav.py` | NAV → Keplerian → ECEF → az/el → polar plot |
| 06 | `06_spp_positioning.py` | single-point positioning from synthetic OBS+NAV |
| 07 | `07_rtk_baseline.py` | `rtk_fix` with LAMBDA on synthetic dual-receiver data |
| 08 | `08_ntrip_to_rtcm.py` | NTRIP byte stream → `rtcm3.iter_messages` (offline replay) |

Most scripts produce numeric output to stdout; the plot scripts
(`01`, `05`) show a matplotlib window unless you pass `--save out.png`.

If you want to use your own files, every script's `MAIN_FILE` constant
at the top can be edited to point at any RINEX/NetCDF/SP3 file.
