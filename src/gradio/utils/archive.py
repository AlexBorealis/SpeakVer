from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def make_zip(
    directory: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Create ZIP archive from a directory.

    Parameters
    ----------
    directory : str | Path
        Directory to archive.

    output_dir : str | Path
        Directory where ZIP archive will be saved.

    Returns
    -------
    Path
        Path to created ZIP archive.
    """

    directory = Path(directory)
    output_dir = Path(output_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a directory.")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_path = output_dir / f"{directory.name}.zip"

    with ZipFile(
        archive_path,
        mode="w",
        compression=ZIP_DEFLATED,
    ) as archive:
        for file in directory.rglob("*"):
            if file.is_file():
                archive.write(
                    filename=file,
                    arcname=file.relative_to(directory.parent),
                )

    return archive_path
