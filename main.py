import sys
import asyncio
import pandas as pd

from gpt_utils import AsyncGPT3ProbBackendLowTemp, AsyncChatQueue

# Run the script with the following arguments:
# python main.py iter1 1979
#                ^^^^^ ^^^^
#          N of iteration
#                     Ref year

async def main():
    iteration = sys.argv[1]
    year = sys.argv[2]
    print(f'Started {year}!')
    dup = pd.read_csv(f"duplets_{year}.csv")
    system = (
        f"In political matters, people talk of 'the Left' and 'the Right'. "
        f"You will be given the names of two parties in their original language, "
        f"and the country of origin of each party. Based on their overall "
        f"ideological stance, which one of these two parties was more "
        f"right-wing in year {year}? Your response can only be exactly either "
        f"'Party 1' or 'Party 2'. "
    )
    users = [(
        f"Party 1: {row['rorigname_1']} "
        f"Party 1 country: {row['rcountry_1']} "
        f"Party 2: {row['rorigname_2']} "
        f"Party 2 country: {row['rcountry_2']} "
    ) for index, row in dup.iterrows()]

    chats = [AsyncGPT3ProbBackendLowTemp(system, user) for user in users]
    chat_queue = AsyncChatQueue(chats, concurrent_n=40)
    res = await chat_queue.get_response()
    txt = [item[0] for item in res]
    probs = [item[1] for item in res]

    dup['gpt3'] = txt
    dup['prob'] = probs

    dup.to_csv(f'classified_{iteration}_{year}.csv', index=False)
    print(f'Finished {year}!')

if __name__ == '__main__':
    asyncio.run(main())
