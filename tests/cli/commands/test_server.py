# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache server`` CLI command."""

# Standard
from unittest.mock import patch
import argparse
import os
import sys

# Third Party
import pytest

# First Party
from lmcache.cli.commands.server import ServerCommand


@pytest.fixture
def cmd():
    return ServerCommand()


@pytest.fixture
def parser(cmd):
    """An ArgumentParser with ServerCommand's arguments registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    cmd.register(sub)
    return p


class TestServerCommandMetadata:
    def test_name(self, cmd):
        assert cmd.name() == "server"

    def test_help(self, cmd):
        assert "server" in cmd.help().lower()


class TestServerCommandArguments:
    def test_registers_subcommand(self, parser):
        """The 'server' subcommand should be parseable."""
        args = parser.parse_args(
            [
                "server",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert hasattr(args, "func")

    def test_mp_server_args_registered(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--host",
                "0.0.0.0",
                "--port",
                "6666",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.host == "0.0.0.0"
        assert args.port == 6666

    def test_http_frontend_args_registered(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--http-host",
                "127.0.0.1",
                "--http-port",
                "9000",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.http_host == "127.0.0.1"
        assert args.http_port == 9000

    def test_prometheus_args_registered(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--prometheus-port",
                "9999",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.prometheus_port == 9999

    def test_default_values(self, parser):
        """Required args only — everything else should get defaults."""
        args = parser.parse_args(
            [
                "server",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.host == "localhost"
        assert args.port == 5555
        assert args.http_host == "0.0.0.0"
        assert args.http_port == 8080
        assert args.cpu_only is False

    def test_cpu_only_flag(self, parser):
        args = parser.parse_args(
            [
                "server",
                "--cpu-only",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.cpu_only is True


class TestServerCommandExecute:
    def test_func_bound_to_execute(self, cmd, parser):
        """parser.parse_args should bind func to ServerCommand.execute."""
        args = parser.parse_args(
            [
                "server",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.func == cmd.execute

    def test_cpu_only_without_full_install_uses_actionable_error(self, cmd, capsys):
        args = argparse.Namespace(cpu_only=True)

        with (
            patch.dict(
                sys.modules,
                {"lmcache.v1.distributed.config": None},
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd.execute(args)

        assert exc_info.value.code == 1
        assert "requires the full lmcache installation" in capsys.readouterr().err

    def test_cpu_only_rejects_non_engine_driven_mode(self, parser):
        http_server = pytest.importorskip("lmcache.v1.multiprocess.http_server")
        with patch.object(http_server, "run_http_server") as mock_run:
            args = parser.parse_args(
                [
                    "server",
                    "--cpu-only",
                    "--l1-size-gb",
                    "4",
                    "--eviction-policy",
                    "LRU",
                ]
            )
            with pytest.raises(
                ValueError,
                match="--cpu-only requires --supported-transfer-mode=engine_driven",
            ):
                ServerCommand().execute(args)

        mock_run.assert_not_called()

    def test_cpu_only_reexecs_with_cuda_hidden(self, parser, monkeypatch):
        http_server = pytest.importorskip("lmcache.v1.multiprocess.http_server")
        args = parser.parse_args(
            [
                "server",
                "--cpu-only",
                "--supported-transfer-mode",
                "engine_driven",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        original_argv = ["/opt/venv/bin/lmcache", "server", "--cpu-only"]
        monkeypatch.setattr(sys, "argv", original_argv)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        with (
            patch.object(http_server, "run_http_server") as mock_run,
            patch(
                "os.execvpe", side_effect=RuntimeError("re-exec requested")
            ) as mock_exec,
            pytest.raises(RuntimeError, match="re-exec requested"),
        ):
            ServerCommand().execute(args)

        mock_run.assert_not_called()
        executable, argv, env = mock_exec.call_args.args
        assert executable == sys.executable
        assert argv == [sys.executable, *original_argv]
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert "CUDA_VISIBLE_DEVICES" not in os.environ

    def test_cpu_only_does_not_reexec_when_cuda_is_already_hidden(
        self, parser, monkeypatch
    ):
        http_server = pytest.importorskip("lmcache.v1.multiprocess.http_server")
        args = parser.parse_args(
            [
                "server",
                "--cpu-only",
                "--supported-transfer-mode",
                "engine_driven",
                "--l1-size-gb",
                "4",
                "--eviction-policy",
                "LRU",
            ]
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

        with (
            patch.object(http_server, "run_http_server") as mock_run,
            patch("os.execvpe") as mock_exec,
        ):
            ServerCommand().execute(args)

        mock_exec.assert_not_called()
        mock_run.assert_called_once()

    def test_execute_calls_run_http_server(self, parser):
        """execute() should call run_http_server with parsed configs."""
        http_server = pytest.importorskip("lmcache.v1.multiprocess.http_server")
        with patch.object(http_server, "run_http_server") as mock_run:
            args = parser.parse_args(
                [
                    "server",
                    "--l1-size-gb",
                    "4",
                    "--eviction-policy",
                    "LRU",
                ]
            )
            cmd = ServerCommand()
            cmd.execute(args)

            mock_run.assert_called_once()
            kwargs = mock_run.call_args.kwargs
            assert "http_config" in kwargs
            assert "mp_config" in kwargs
            assert "storage_manager_config" in kwargs
            assert "obs_config" in kwargs
