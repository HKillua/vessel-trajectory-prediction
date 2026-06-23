#!/bin/bash
# download_noaa_ais.sh - 下载 NOAA AIS 数据集 (2024年6月-9月)
# 用法: bash download_noaa_ais.sh [start_date] [end_date]
#
# 示例:
#   bash download_noaa_ais.sh              # 下载 2024-06-01 到 2024-09-30
#   bash download_noaa_ais.sh 2024-07-01 2024-07-31  # 只下载7月

SAVE_DIR="$(cd "$(dirname "$0")" && pwd)"
START_DATE="${1:-2024-06-01}"
END_DATE="${2:-2024-09-30}"
BASE_URL="https://coast.noaa.gov/htdata/CMSP/AISDataHandler"

echo "============================================"
echo "  NOAA AIS Data Downloader"
echo "  Date Range: $START_DATE to $END_DATE"
echo "  Save Dir:   $SAVE_DIR"
echo "============================================"

# 计算总天数
START_TS=$(date -d "$START_DATE" +%s)
END_TS=$(date -d "$END_DATE" +%s)
TOTAL_DAYS=$(( (END_TS - START_TS) / 86400 + 1 ))
echo "  Total days to download: $TOTAL_DAYS"
echo ""

# 创建子目录
mkdir -p "$SAVE_DIR/2024" "$SAVE_DIR/2025"

CURRENT_DATE="$START_DATE"
DAY_NUM=0
SUCCESS=0
FAIL=0
SKIP=0

while true; do
    # 检查是否超过结束日期
    CUR_TS=$(date -d "$CURRENT_DATE" +%s)
    if [ "$CUR_TS" -gt "$END_TS" ]; then
        break
    fi
    
    DAY_NUM=$((DAY_NUM + 1))
    YEAR=$(date -d "$CURRENT_DATE" +%Y)
    MONTH=$(date -d "$CURRENT_DATE" +%m)
    DAY=$(date -d "$CURRENT_DATE" +%d)
    
    # 构建文件名和URL
    FILENAME="AIS_${YEAR}_${MONTH}_${DAY}.zip"
    URL="${BASE_URL}/${YEAR}/${FILENAME}"
    SAVE_PATH="${SAVE_DIR}/${YEAR}/${FILENAME}"
    
    # 如果已存在则跳过
    if [ -f "$SAVE_PATH" ]; then
        echo "[$DAY_NUM/$TOTAL_DAYS] $FILENAME already exists, skipping"
        SKIP=$((SKIP + 1))
        CURRENT_DATE=$(date -d "$CURRENT_DATE + 1 day" +%Y-%m-%d)
        continue
    fi
    
    # 下载
    echo -n "[$DAY_NUM/$TOTAL_DAYS] Downloading $FILENAME ... "
    if wget -q --timeout=60 --tries=2 -O "$SAVE_PATH" "$URL" 2>/dev/null; then
        FILE_SIZE=$(du -h "$SAVE_PATH" | cut -f1)
        echo "OK ($FILE_SIZE)"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAILED (file may not exist)"
        rm -f "$SAVE_PATH"
        FAIL=$((FAIL + 1))
    fi
    
    CURRENT_DATE=$(date -d "$CURRENT_DATE + 1 day" +%Y-%m-%d)
done

echo ""
echo "============================================"
echo "  Download Complete!"
echo "  Success: $SUCCESS"
echo "  Failed:  $FAIL"
echo "  Skipped: $SKIP"
echo "  Total:   $TOTAL_DAYS"
echo ""
echo "  Total size:"
du -sh "$SAVE_DIR"/20*/  2>/dev/null
echo "============================================"
