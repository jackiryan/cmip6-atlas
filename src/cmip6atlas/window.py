"""
Copyright (c) 2025 Jacqueline Ryan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import dask
import dask.delayed
from dask.distributed import Client

from cmip6atlas.cli import get_parser
from cmip6atlas.download import download_granules
from cmip6atlas.ensemble import ensemble_process


def process_window(
    variable: str,
    scenario: str,
    start_year: int,
    end_year: int,
    input_dir: str,
    output_dir: str,
    models: list[str] | None = None,
    exclude_models: list[str] | None = None,
    max_workers: int = 5,
) -> None:
    client = Client()
    print(f"Dask dashboard available at: {client.dashboard_link}")

    def year_task(year: int):
        download_granules(
            variable,
            scenario,
            year,
            output_dir=input_dir,
            models=models,
            exclude_models=exclude_models,
            max_workers=1,
        )
        ensemble_process(input_dir, output_dir, variable, scenario, year)

    tasks: list[dask.delayed.Delayed] = []
    for year in range(start_year, end_year + 1):
        task = dask.delayed(year_task)(year)
        tasks.append(task)

    dask.compute(*tasks)

    client.close()


def cli() -> None:
    """
    Command Line Interface for downloading files independently of
    other processing steps.
    """
    parser = get_parser(
        "Generate an ensemble mean (average model output) over a time window, including file downloads.",
        task="window",
    )

    args = parser.parse_args()
    process_window(
        args.variable,
        args.scenario,
        args.start_year,
        args.end_year,
        args.input_dir,
        args.output_dir,
        args.models,
        args.exclude_models,
        args.max_workers,
    )
