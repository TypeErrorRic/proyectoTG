"""
Conexion simple a MariaDB/MySQL usando PyMySQL y consulta de prueba.

Configura los siguientes env vars o ajusta los valores por defecto:
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
"""

import os
from typing import Any, Dict

import pymysql
from pymysql.cursors import DictCursor


def get_db_config() -> Dict[str, Any]:
    """Lee configuracion de conexion desde variables de entorno."""
    return {
        "host": os.getenv("DB_HOST", "192.168.1.7"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "jetson"),
        "password": os.getenv("DB_PASSWORD", "1234"),
        "database": os.getenv("DB_NAME", "midb"),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
    }


def get_connection():
    """Devuelve una conexion PyMySQL lista para usar."""
    cfg = get_db_config()
    return pymysql.connect(**cfg)


def test_query() -> Dict[str, Any]:
    """
    Ejecuta una consulta de prueba para validar la conexion.

    Devuelve un dict con los campos devueltos por la query.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DATABASE() AS db, NOW() AS ahora")
            return cur.fetchone() or {}
