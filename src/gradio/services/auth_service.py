import os


class AuthService:
    """
    Service responsible for loading Gradio authentication users.
    """

    @staticmethod
    def get_users() -> list[tuple[str, str]]:
        """
        Load users from environment variables.

        Expected variables:

            USER1_LOGIN
            USER1_PASSWORD

            ...

        Returns
        -------
        list[tuple[str, str]]
            List of (login, password) tuples.
        """

        users = []
        index = 1

        while True:
            login = os.getenv(f"USER{index}_LOGIN")
            password = os.getenv(f"USER{index}_PASSWORD")

            if login is None or password is None:
                break

            users.append(
                (
                    login,
                    password,
                )
            )

            index += 1

        return users

    @staticmethod
    def validate() -> list[tuple[str, str]]:
        """
        Validate that at least one user is configured.
        """

        users = AuthService.get_users()

        if not users:
            raise RuntimeError("No users configured. Check your .env file.")

        return users
