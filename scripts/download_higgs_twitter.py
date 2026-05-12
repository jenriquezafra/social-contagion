"""Download the SNAP Higgs Twitter files used by the Streamlit app."""

from __future__ import annotations

from src.twitter_higgs import RAW_DATA_URLS, download_higgs_data


def main() -> None:
    downloaded = download_higgs_data()
    for path in downloaded:
        url = RAW_DATA_URLS[path]
        print(f"Downloaded {url} -> {path}")


if __name__ == "__main__":
    main()
