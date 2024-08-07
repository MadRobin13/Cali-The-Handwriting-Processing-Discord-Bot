from ultralytics import YOLO
import discord
import dotenv
import os

model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

async def main():
    dotenv.load_dotenv();
    await client.run(os.getenv("TOKEN"));

main()

@client.event

async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.attachments:
        for attachment in message.attachments:
            await attachment.save(attachment.filename)
            results = model(attachment.filename)
            message.channel.send(results)