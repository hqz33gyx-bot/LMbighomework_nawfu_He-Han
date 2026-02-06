# 获取脚本所在目录并进入
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 默认配置
DEFAULT_PORT=9198
DEFAULT_HOST="0.0.0.0"
DEV_MODE=0

# 解析命令行参数（仅保留核心参数解析，移除help的echo/exit）
while [[ $# -gt 0 ]]; do
    case $1 in
        --port|-p)
            TRAINER_PORT="$2"
            shift 2
            ;;
        --host|-H)
            TRAINER_HOST="$2"
            shift 2
            ;;
        --dev|-d)
            DEV_MODE=1
            shift
            ;;
        --help|-h)
            exit 0
            ;;
        *)
            exit 1
            ;;
    esac
done

# 加载 .env 配置
set -a
source "$SCRIPT_DIR/.env"
set +a

# 应用默认值并导出环境变量
export TRAINER_PORT=${TRAINER_PORT:-$DEFAULT_PORT}
export TRAINER_HOST=${TRAINER_HOST:-$DEFAULT_HOST}
export MODEL_PATH=${MODEL_PATH:-"$SCRIPT_DIR/models"}
export DATASET_PATH=${DATASET_PATH:-"$SCRIPT_DIR/datasets"}
export LORA_PATH=${LORA_PATH:-"$SCRIPT_DIR/output"}
export OLLAMA_HOST=${OLLAMA_HOST:-"http://127.0.0.1:11434"}
export OLLAMA_MODEL=${OLLAMA_MODEL:-"llama3.2-vision"}

# 创建必要目录
mkdir -p "$DATASET_PATH" "$LORA_PATH" logs

# 激活虚拟环境
VENV_DIR="$SCRIPT_DIR/venv"
source "$VENV_DIR/bin/activate"

# 设置 Python 路径
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# 前端构建（直接执行构建命令，移除判断逻辑）
cd "$SCRIPT_DIR/webui-vue"
npm run build --silent
cd "$SCRIPT_DIR"

# 获取本机IP（仅保留Linux命令）
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')

# 启动服务（按开发模式/生产模式执行核心命令）
cd "$SCRIPT_DIR/webui-vue/api"
if [ "$DEV_MODE" -eq 1 ]; then
    python -m uvicorn main:app --host "$TRAINER_HOST" --port "$TRAINER_PORT" --reload --reload-dir "$SCRIPT_DIR/webui-vue/api" --log-level info
else
    python -m uvicorn main:app --host "$TRAINER_HOST" --port "$TRAINER_PORT" --log-level warning
fi