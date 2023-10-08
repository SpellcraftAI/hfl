.PHONY: all load_env_win_cmd load_env_win_pow create_env_win_cmd create_env_win_pow install run

.DEFAULT_GOAL := all

all: load_env_win_cmd install run

load_env_win_cmd:
	.env\Scripts\activate

load_env_win_pow:
	.env\Scripts\Activate.ps1

create_env_win_cmd:
	python -m venv .env

create_env_win_pow:
	python -m venv .env
	Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

install:
	pip install -r requirements.txt

run:
	@python src/__init__.py
