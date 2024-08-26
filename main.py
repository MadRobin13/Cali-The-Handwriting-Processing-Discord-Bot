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
    if message.content == "ping":
        await message.channel.send("pong")

    if message.attachments:
        for attachment in message.attachments:

            img_path = os.path.join("./images", attachment.filename)
            await attachment.save(img_path)

            result = model(img_path, save=True)
            print(result)

            # result_img_path = img_path.replace(".jpg", "_result.jpg")
            # result[0].plot(save=True, path=result_img_path)

            result_img_path = os.path.join("runs\detect\predict", attachment.filename)

            with open(result_img_path) as f:
                send_img = discord.File(f)
                await message.channel.send(file=send_img)

            os.remove(result_img_path)

def main():
    client.run(os.getenv("TOKEN"));

if __name__ == "__main__":
    main()