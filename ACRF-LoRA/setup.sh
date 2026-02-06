#!/bin/bash
set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"


# 安装本项目
pip install -e . -q

# 创建 .env 文件
if [ ! -f ".env" ]; then
    cp env.example .env
fi

# 检查 Node.js

NODE_VERSION=$(node --version)
NODE_MAJOR_VERSION=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')


# 构建前端
cd "$SCRIPT_DIR/webui-vue"

if [ ! -d "node_modules" ]; then
    npm install --silent
fi

npm run build --silent

cd "$SCRIPT_DIR"
