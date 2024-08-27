import asyncio
from ultralytics import YOLO
import discord
from dotenv import load_dotenv
import os
import PIL

model = YOLO("yolov8s.pt")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

load_dotenv();

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.attachments:
        for attachment in message.attachments:

            img_path = os.path.join("./images", attachment.filename)
            await attachment.save(img_path)

            result = model(img_path, save=True, project="results", name=attachment.filename)

            result_img_path = os.path.join("results\\" + attachment.filename, attachment.filename)
            print(result_img_path)

            with open(result_img_path, "rb") as f:
                send_img = discord.File(f)
                await message.channel.send(file=send_img)
                # await message.channel.send(str(result[0]))
                for det in result[0].detections:
                    x1, y1, x2, y2 = det.box.xyxy  # Bounding box coordinates
                    conf = det.conf  # Confidence
                    cls = det.cls  # Detected class ID
                    await message.channel.send(f"Coordinates: {x1}, {y1}, {x2}, {y2}, Conf: {conf}, Class: {cls}")

def main():
    client.run(os.getenv("TOKEN"));

if __name__ == "__main__":
    main()