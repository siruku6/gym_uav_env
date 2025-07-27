# Docker での環境構築

流れを以下に記載する

## 1. Python 環境の準備

- docker コンテナの build

    ```bash
    $ cd environments/uav_env
    $ cp .env.example .env

    $ docker compose build
    ```

これにより、 docker で jupyter と tensorboard を起動できるようになるので、実施しておくことを推奨

## 2. 学習〜評価

    ```bash
    $ docker compose up
    $ docker compose exec uav_env bash
    ```
