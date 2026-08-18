#!/bin/bash
# 装配 GitHub 英文仓库：从当前项目生成 en_release/（英文 README + 英文 docs/ + 代码/数据）
# 中文文档留在 Gitee 版，不进 GitHub 英文仓库。
ROOT="/root/Default Project"
OUT="$ROOT/en_release"
rm -rf "$OUT" && mkdir -p "$OUT/docs"

# 1) 复制代码/数据/实验（排除 .git、docs 中文、docs-en、en_release、dist 大二进制可选）
cd "$ROOT"
for item in $(ls -A | grep -vE '^(\.git|docs|docs-en|en_release|dist|archive/scripts_20260816)$'); do
  cp -r "$item" "$OUT/" 2>/dev/null
done

# 2) 英文文档 → docs/
for f in docs-en/*.md; do
  b=$(basename "$f")
  [ "$b" = "README.md" ] && continue
  cp "$f" "$OUT/docs/"
done

# 3) 英文根 README
[ -f docs-en/README.md ] && cp docs-en/README.md "$OUT/README.md"

# 3.5) 复制 .gitignore（排除构建产物/缓存）
cp "$ROOT/.gitignore" "$OUT/.gitignore" 2>/dev/null

# 4) 清理根级符号链接（英文仓库根只留 README/LICENSE/目录）
find "$OUT" -maxdepth 1 -type l -delete 2>/dev/null

# 4.5) 清理嵌套 mnist_data 绝对符号链接（数据本体在 en_release/mnist_data）
find "$OUT" -path "*/mnist_data" -type l -delete 2>/dev/null

# 4.6) 子目录 README 英文化（模板在 docs-en/subreadmes/，覆盖中文版）
cp -r "$ROOT/docs-en/subreadmes/." "$OUT/" 2>/dev/null

# 5) 清理运行残留
find "$OUT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find "$OUT" -name "*.pyc" -delete 2>/dev/null

echo "==> GitHub 英文仓库骨架: $OUT"
echo "   README.md: $([ -f "$OUT/README.md" ] && echo OK || echo 缺失)"
echo "   docs 文件数: $(ls "$OUT/docs" | wc -l)"
echo "   总文件数: $(find "$OUT" -type f | wc -l)"
