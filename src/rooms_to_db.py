import json
from uuid import uuid4

def rooms_to_db(DB, rooms_file):
    with open(rooms_file, "r") as f:
        rooms = json.loads(f.read())
        for room in rooms:
            room_uuid = str(uuid4())
            
            DB.execute("""INSERT INTO rooms VALUES (?, ?)""", (room_uuid, room["name"],))
            for shelf in room["shelves"]:
                shelf_uuid = str(uuid4())
                name = shelf["name"]
                width = shelf["width"]
                room = room_uuid
                DB.execute("""INSERT INTO shelves VALUES (?, ?, ?, ?, ?, ?)""", (shelf_uuid, name, width, room_uuid, 0, 0))
