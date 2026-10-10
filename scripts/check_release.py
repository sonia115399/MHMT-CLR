import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {".gitignore", "LICENSE", "README.md", "requirements.txt",
              "requirements-modern.txt", "requirements-dev.txt", "requirements-original.txt", "pyproject.toml",
              "code/README.md", "SimCLR/README.md"}
EXTENSIONS = {"mhmt": {".py"}, "code": {".py"}, "SimCLR": {".py"},
              "scripts": {".py"}, "tests": {".py"}, "configs": {".json"}}


def main():
    result = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                            check=True, capture_output=True)
    names = [name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    size = 0
    for name in names:
        path = Path(name)
        allowed = name in ROOT_FILES or (len(path.parts) > 1 and path.parts[0] in EXTENSIONS
                                         and path.suffix in EXTENSIONS[path.parts[0]])
        if not allowed:
            raise ValueError(f"待发布文件类型或位置异常：{name}")
        payload = (ROOT / path).read_bytes()
        if payload.startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"文件包含 BOM：{name}")
        text = payload.decode("utf-8")
        if path.suffix == ".py":
            ast.parse(text, filename=name, feature_version=(3, 8))
        if re.search(r"/(?:Users|home)/", text):
            raise ValueError(f"文件仍含本机或历史服务器绝对路径：{name}")
        size += len(payload)

    examples = ["data/example.jpg", "outputs/run/metrics.json", "code/log/model.pth",
                "SimCLR/runs/source.pth", "code/multi_data/corn/train/image.jpg",
                "SimCLR/tomato/train/image.JPG", "MHMT_CLR_Highlighted.pdf",
                ".local_archive/code/run.py", ".serena/memories/local.md", ".env",
                "docs/manuscript_consistency_audit.md", "docs/original_sources.csv"]
    for name in examples:
        check = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", name])
        if check.returncode != 0:
            raise ValueError(f"发布忽略规则未覆盖：{name}")
    print(f"发布清单通过：{len(names)} 个 UTF-8 文本文件，共 {size:,} 字节；数据和产物已排除。")


if __name__ == "__main__":
    main()
