import logging

def configure_logging(level=logging.INFO, suppress_http=True):
    """
    Configure logging for the Poet system.
    
    Args:
        level: Logging level (default: INFO)
        suppress_http: Whether to suppress HTTP request logs (default: True)
    """
    # Basic logging configuration
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if suppress_http:
        # Suppress HTTP request logs from various libraries
        http_loggers = [
            "httpx",           # httpx library
            "requests",        # requests library  
            "urllib3",         # urllib3 library
            "httpcore",        # httpcore library
            "openai",          # OpenAI client
            "anthropic",       # Anthropic client
            "groq",           # Groq client
        ]
        
        for logger_name in http_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        # Also suppress some verbose loggers
        verbose_loggers = [
            "httpx._client",
            "httpx._transport",
            "openai._client",
            "anthropic._client",
        ]
        
        for logger_name in verbose_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Set root logger level
    logging.getLogger().setLevel(level)
    
    return logging.getLogger(__name__)
