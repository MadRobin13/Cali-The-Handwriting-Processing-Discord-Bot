from ultralytics import YOLO
import discord
from dotenv import load_dotenv
import os

model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

load_dotenv();

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.attachments:
        for attachment in message.attachments:

            await attachment.save(attachment.filename)
            result = model(attachment.filename)

            # testing stuff

            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk

            ###############


            print(result)
            await message.channel.send(result.plot())

def main():
    client.run(os.getenv("TOKEN"));

if __name__ == "__main__":
    main()