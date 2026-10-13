import datetime
from zoneinfo import ZoneInfo

BRASILIA_TZ = ZoneInfo("America/Sao_Paulo")


def get_brasilia_time():
    """Retorna objeto datetime atual no fuso de Brasília."""
    return datetime.datetime.now(BRASILIA_TZ)
