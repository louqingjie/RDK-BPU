# syntax=docker/dockerfile:1.4
# 基于 OE v1.2.8 GPU 镜像，加装 sshd 供 Trae Remote-SSH 连接
FROM openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8

# 安装 OpenSSH 服务端，允许 root 登录
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd \
 && sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config \
 && echo 'root:rdk2024' | chpasswd

# 可选：构建时传入宿主机公钥实现免密登录（--build-arg SSH_PUB_KEY=...）
ARG SSH_PUB_KEY=""
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh \
 && if [ -n "$SSH_PUB_KEY" ]; then \
        echo "$SSH_PUB_KEY" > /root/.ssh/authorized_keys \
        && chmod 600 /root/.ssh/authorized_keys; \
    fi

# 入口：先执行工程自检脚本（复用 post_create.sh），再前台启动 sshd 保持容器存活
# 用 printf 生成，避免 COPY 带入 Windows CRLF 导致 bad interpreter
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'if [ -f /workspace/.devcontainer/post_create.sh ]; then' \
    '  bash /workspace/.devcontainer/post_create.sh || true' \
    'fi' \
    'exec /usr/sbin/sshd -D' \
    > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/entrypoint.sh"]
