## Notifications
Contains code to send notifications to Discord or Pushover. Used as a kind of `print` statement.

### Discord Notifications
1. Create a Discord account.
2. Create a server.
3. Add a webhook to a channel of the server.
4. Add the webhook url to the `api_config.py` file in the root directory:
```
DISCORD_WEBHOOK_URL = '<your webhook url>'
```
5. Download the Discord app.

### Pushover Notifications
> This method costs money.
1. Create a Pushover account.
2. Create an application.
3. Add these two lines to the `api_config.py` file in the root directory:
```
PUSHOVER_APP_KEY = '<your app key>'
PUSHOVER_USER_KEY = '<your user key>'
```
4. Download the Pushover app.