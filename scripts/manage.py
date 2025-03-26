import os
import sys


current_path = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(current_path, "..")
sys.path.insert(0, repo_root)

def main():
    """Run administrative tasks."""
    os.environ.setdefault(
        'DJANGO_SETTINGS_MODULE', 'galapassistant.settings.settings'
    )
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Ensure it is installed and available on your PYTHONPATH."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
