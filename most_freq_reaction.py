import pandas as pd
from ast import literal_eval


# Define a function to extract the most frequent emoticon
def get_most_frequent_reaction(reactions_list):
    max_count = 0
    most_frequent_reaction = None

    for reaction_dict in reactions_list:
        for emoticon, count in reaction_dict.items():
            if count > max_count:
                max_count = count
                most_frequent_reaction = emoticon

    return most_frequent_reaction


# def filter_reactions_by_biggest_num(reactions_object):
#     if len(reactions_object["results"]) != 0:
#         max_react = max(reactions_object["results"], key=lambda x: x['count'])
#         if 'emoticon' in max_react["reaction"]:
#             return max_react['reaction']['emoticon']
#         else:
#             return max_react['reaction']['document_id']
#     else:
#         return 'No reaction'
    # {'_': 'MessageReactions', 'results': [
    #     {'_': 'ReactionCount', 'reaction': {'_': 'ReactionEmoji', 'emoticon': '🤡'}, 'count': 27, 'chosen_order': None},
    #     {'_': 'ReactionCount', 'reaction': {'_': 'ReactionEmoji', 'emoticon': '😁'}, 'count': 10, 'chosen_order': None},
    #     {'_': 'ReactionCount', 'reaction': {'_': 'ReactionEmoji', 'emoticon': '🥰'}, 'count': 2, 'chosen_order': None},
    #     {'_': 'ReactionCount', 'reaction': {'_': 'ReactionEmoji', 'emoticon': '😢'}, 'count': 2, 'chosen_order': None},
    #     {'_': 'ReactionCount', 'reaction': {'_': 'ReactionEmoji', 'emoticon': '🤔'}, 'count': 1, 'chosen_order': None}
    #     ],
    #  'min': False, 'can_see_list': False, 'recent_reactions': []}


# input_file = 'oldlentach_messages_cleaned.csv'
# output_file = 'oldlentach_most_freq_reactions.csv'

input_file = 'data/svtvnews_messages_cleaned.csv'
output_file = 'data/svtvnewsw_most_freq_reactions.csv'

# read csv dat
df = pd.read_csv(input_file)
df['reactions'] = df['reactions'].apply(literal_eval)

# Apply the function to the "reactions" column
df['reactions'] = df['reactions'].apply(get_most_frequent_reaction)
df = df.rename(columns={'reactions': 'most_freq_reaction'})
print(df.head())

df.to_csv(output_file, index=False)