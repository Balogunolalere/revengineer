"""
Docker Attack Lab — isolated execution environment for security tools.

Creates a purpose-built Docker container with all offensive tools
pre-installed. Tools execute inside this container, isolated from
the host system.

Features:
  - Pre-built image with nmap, nuclei, subfinder, httpx, nikto, etc.
  - Network isolation (dedicated Docker network)
  - Resource limits (memory, CPU)
  - Cap-drop ALL + only NET_RAW for nmap SYN
  - Evidence volume mount
  - Auto-cleanup on exit
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from cookbook.arsenal.config import ArsenalConfig

logger = logging.getLogger("arsenal.lab")


DOCKERFILE = r"""
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git unzip jq \
    dnsutils whois net-tools iputils-ping traceroute \
    nmap nikto \
    python3 python3-pip \
    golang-go \
    && rm -rf /var/lib/apt/lists/*

# Nuclei
RUN go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest && \
    mv /root/go/bin/nuclei /usr/local/bin/

# Subfinder
RUN go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest && \
    mv /root/go/bin/subfinder /usr/local/bin/

# HTTPX
RUN go install -v github.com/projectdiscovery/httpx/cmd/httpx@latest && \
    mv /root/go/bin/httpx /usr/local/bin/

# FFUF
RUN go install github.com/ffuf/ffuf/v2@latest && \
    mv /root/go/bin/ffuf /usr/local/bin/

# WhatWeb
RUN apt-get update && apt-get install -y --no-install-recommends whatweb && \
    rm -rf /var/lib/apt/lists/*

# TestSSL
RUN git clone --depth 1 https://github.com/drwetter/testssl.sh.git /opt/testssl && \
    ln -s /opt/testssl/testssl.sh /usr/local/bin/testssl.sh

# Wordlists
RUN mkdir -p /usr/share/wordlists/dirb && \
    wget -q -O /usr/share/wordlists/dirb/common.txt \
    https://raw.githubusercontent.com/v0re/dirb/master/wordlists/common.txt 2>/dev/null || \
    echo "/ /admin /api /login /wp-admin /wp-login.php /.env /.git /robots.txt /sitemap.xml" \
    | tr ' ' '\n' > /usr/share/wordlists/dirb/common.txt

# Update nuclei templates
RUN nuclei -update-templates 2>/dev/null || true

# Clean up Go build cache
RUN rm -rf /root/go/pkg /root/go/src /root/.cache/go-build

# Evidence directory
RUN mkdir -p /evidence

WORKDIR /workspace

# Non-root user for non-privileged scans
RUN useradd -m -s /bin/bash arsenal
# But keep root for SYN scans (controlled via cap-drop)

ENTRYPOINT ["/bin/bash", "-c"]
"""

COMPOSE_TEMPLATE = """
services:
  arsenal-lab:
    build:
      context: .
      dockerfile: Dockerfile
    image: {image}
    container_name: arsenal-lab
    networks:
      - {network}
    cap_drop:
      - ALL
    cap_add:
      - NET_RAW
    security_opt:
      - no-new-privileges
    mem_limit: {memory}
    cpus: {cpus}
    volumes:
      - {evidence_dir}:/evidence
    read_only: false
    restart: "no"

networks:
  {network}:
    driver: bridge
"""


class AttackLab:
    """
    Manages the Docker attack lab environment.

    Usage:
        lab = AttackLab(config)
        await lab.setup()     # Build image + create network
        # ... run tools via ToolExecutor (auto routes to Docker)
        await lab.teardown()  # Clean up
    """

    def __init__(self, config: ArsenalConfig):
        self.config = config
        self._lab_dir: Path | None = None
        self._built = False
        self._network_created = False

    async def setup(self) -> None:
        """Build the Docker image and create the network."""
        if not shutil.which("docker"):
            raise RuntimeError(
                "Docker is not installed or not in PATH. "
                "Install Docker or set use_docker=False in config."
            )

        # Create lab directory
        self._lab_dir = Path(self.config.output_dir or ".") / ".arsenal-lab"
        self._lab_dir.mkdir(parents=True, exist_ok=True)

        # Write Dockerfile
        dockerfile = self._lab_dir / "Dockerfile"
        dockerfile.write_text(DOCKERFILE)

        # Create evidence directory
        evidence_dir = self.config.get_evidence_dir()
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # Build image
        logger.info(f"Building Docker image: {self.config.docker_image}")
        await self._run_docker(
            f"docker build -t {self.config.docker_image} {self._lab_dir}",
            timeout=600,  # building can take a while
        )
        self._built = True
        logger.info("Docker image built successfully")

        # Create network
        await self._run_docker(
            f"docker network create {self.config.docker_network} 2>/dev/null || true",
            timeout=30,
        )
        self._network_created = True
        logger.info(f"Docker network ready: {self.config.docker_network}")

    async def teardown(self) -> None:
        """Clean up the Docker environment."""
        try:
            # Stop and remove any running lab containers
            await self._run_docker(
                "docker ps -q --filter name=arsenal-lab | xargs -r docker stop",
                timeout=30,
            )
            await self._run_docker(
                "docker ps -aq --filter name=arsenal-lab | xargs -r docker rm",
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"Container cleanup: {e}")

        try:
            if self._network_created:
                await self._run_docker(
                    f"docker network rm {self.config.docker_network} 2>/dev/null || true",
                    timeout=15,
                )
        except Exception as e:
            logger.warning(f"Network cleanup: {e}")

    async def is_ready(self) -> bool:
        """Check if the Docker lab is ready."""
        try:
            proc = await asyncio.create_subprocess_shell(
                f"docker image inspect {self.config.docker_image} > /dev/null 2>&1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            return False

    async def check_tools(self) -> dict[str, bool]:
        """Check which tools are available in the Docker image."""
        tools_to_check = [
            "nmap", "nuclei", "subfinder", "httpx", "ffuf",
            "nikto", "whatweb", "testssl.sh", "dig", "whois", "curl",
        ]
        results = {}
        for tool in tools_to_check:
            try:
                stdout, _, code = await self._exec_in_container(f"which {tool}")
                results[tool] = (code == 0)
            except Exception:
                results[tool] = False
        return results

    async def _exec_in_container(
        self, command: str, timeout: float = 30,
    ) -> tuple[str, str, int]:
        """Execute a command in a one-shot container."""
        import shlex
        docker_cmd = (
            f"docker run --rm "
            f"--network {self.config.docker_network} "
            f"--memory {self.config.docker_memory} "
            f"--cpus {self.config.docker_cpus} "
            f"--security-opt no-new-privileges "
            f"--cap-drop ALL --cap-add NET_RAW "
            f"{self.config.docker_image} "
            f"{shlex.quote(command)}"
        )

        proc = await asyncio.create_subprocess_shell(
            docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
        return (
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            proc.returncode or 0,
        )

    async def _run_docker(self, command: str, timeout: float = 60) -> str:
        """Run a docker management command."""
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
        if proc.returncode != 0:
            err = stderr.decode(errors="replace")
            # Don't raise on non-fatal errors
            if "already exists" not in err and "No such" not in err:
                logger.warning(f"Docker command warning: {err[:200]}")
        return stdout.decode(errors="replace")

    def write_compose(self) -> Path:
        """Write a docker-compose.yml for manual use."""
        evidence_dir = self.config.get_evidence_dir()
        evidence_dir.mkdir(parents=True, exist_ok=True)

        compose = COMPOSE_TEMPLATE.format(
            image=self.config.docker_image,
            network=self.config.docker_network,
            memory=self.config.docker_memory,
            cpus=self.config.docker_cpus,
            evidence_dir=str(evidence_dir.absolute()),
        )

        lab_dir = Path(self.config.output_dir or ".") / ".arsenal-lab"
        lab_dir.mkdir(parents=True, exist_ok=True)
        compose_path = lab_dir / "docker-compose.yml"
        compose_path.write_text(compose)
        logger.info(f"Compose file written: {compose_path}")
        return compose_path
