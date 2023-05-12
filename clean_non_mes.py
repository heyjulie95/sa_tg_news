import json
import pandas as pd

# input_file = "oldlentach_channel_messages.json"
# output_file = "oldlentach_messages_cleaned.csv"

input_file = "data/svtvnews_channel_messages.json"
output_file = "data/svtvnews_messages_cleaned.csv"

property_name = "_"
property_value = "Message"
properties_to_remove_from_messages = ["_", "id", "peer_id", "date", "out", "media_unread", "silent", "post", "mentioned", "from_scheduled",
                        "legacy", "pinned",
                        "edit_hide", "noforwards", "from_id", "fwd_from", "via_bot_id", "reply_to", "media",
                        "reply_markup", "entities", "views", "forwards", "replies", "edit_date", "post_author",
                        "grouped_id", "restriction_reason", "ttl_period"]


def remove_properties(data, properties_to_remove):
    for item in data:
        for prop in properties_to_remove:
            item.pop(prop, None)


def clean_dataset_from_non_messages(input_file, output_file, property_name, property_value,
                                    properties_to_remove_from_messages):
    with open(input_file, "r") as f:
        data = json.load(f)

    # Filter out objects that don't have the specified property value
    data = [item for item in data if item.get(property_name) == property_value and item.get("message") != ''] # here we take only Messages, obj-s that have "_": Message prop
    messages_having_reactions = [item for item in data if item.get("reactions")]
    remove_properties(messages_having_reactions, properties_to_remove_from_messages)
    # making a pretty list of reactions
    for i in messages_having_reactions:
        i["reactions"] = i["reactions"]["results"]
        for item in i["reactions"]:
            if 'emoticon' in item['reaction']:
                emoticon = item['reaction']['emoticon']
            else:
                emoticon = item['reaction']['document_id']
            count = item['count']
            item.clear()
            item[emoticon] = count

    df = pd.DataFrame(messages_having_reactions, columns=["message", "reactions"])
    df.to_csv(output_file, index=False)


clean_dataset_from_non_messages(input_file, output_file, property_name, property_value,
                                properties_to_remove_from_messages)