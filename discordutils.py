from discord_webhook import DiscordWebhook

webhook_url = ''  # replace with your webhook's URL


def send_msg(content):
    """
       Send a message to a Discord webhook.

       Args:
           content: The content of the message.
       """
    webhook = DiscordWebhook(url=webhook_url, content=content)
    webhook.execute()
