import datetime
import pytz

def get_brasilia_time():
    """Retorna objeto datetime atual no fuso de Brasília."""
    brasilia_tz = pytz.timezone('America/Sao_Paulo')
    return datetime.datetime.now(brasilia_tz)
