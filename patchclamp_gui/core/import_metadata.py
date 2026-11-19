import csv
import os
from pathlib import Path


HEADERS = ["记录标识", "文件名", "分组", "备注"]


def scan(folder, project):
    folder = Path(folder).resolve()
    excluded = {Path(project).resolve() / name for name in ("raw", "results")}
    for directory, children, files in os.walk(folder):
        parent = Path(directory)
        if parent in excluded or any(root in parent.parents for root in excluded):
            children.clear()
            continue
        children[:] = sorted(name for name in children if not (parent / name).is_symlink() and not (parent / name / "project.pca.h5").is_file())
        for name in sorted(files):
            path = parent / name
            if not path.is_symlink() and name.lower() != "project.pca.h5" and path.suffix.lower() in {".abf", ".h5", ".hdf5", ".npy"}:
                yield path, (parent.relative_to(folder).as_posix() if parent != folder else folder.name)


def template(path, root, cells, importing):
    known = {Path(cell.rec.path).relative_to(root).as_posix(): cell for cell in cells if cell.rec.path}
    if not known:
        raise ValueError("项目中没有已导入的记录。")
    if not importing:
        with open(path, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADERS)
            for key, cell in known.items():
                writer.writerow([key, Path(key).name, cell.group, cell.note])
        return len(known)
    changes = {}
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        if sorted(reader.fieldnames or []) != sorted(HEADERS):
            raise ValueError("表头无效，请使用本项目的 UTF-8 CSV 模板。")
        for row in reader:
            key = row["记录标识"]
            if None in row or None in row.values() or key not in known or key in changes or row["文件名"] != Path(key).name:
                raise ValueError(f"第 {reader.line_num} 行重复或不匹配，未应用任何修改。")
            changes[key] = row["分组"].strip(), row["备注"]
    if not changes:
        raise ValueError("模板没有记录。")
    for key, values in changes.items():
        known[key].group, known[key].note = values
    return len(changes)
