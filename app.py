from src.config import (
    SERVER_NAME,
    SERVER_PORT,
)
from src.gradio.services.auth_service import AuthService
from src.gradio.ui.layout import create_app


def main() -> None:
    """
    Application entry point.
    """
    demo = create_app()

    demo.queue(max_size=3)

    demo.launch(
        auth=AuthService.validate(),
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
    )


if __name__ == "__main__":
    main()
