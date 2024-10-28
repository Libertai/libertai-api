from src.account_manager import AccountManager
from src.interfaces.account import TokenAccount

account_manager = AccountManager()


def add_application_task(account: TokenAccount):
    print("application task called")
    account_manager.add_account(account)


def call_event_task(token):
    account_manager.increment_calls(token)
    print("collect calls...")
