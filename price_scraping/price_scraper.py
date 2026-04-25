import csv
import requests
from http.client import RemoteDisconnected
import re
import time
from urllib3.exceptions import ProtocolError, HTTPError

# checks for prices in USD, default sort of results ()
ENDPOINT = "https://boardgameprices.com/api/info?currency=USD&destination=US&sitename=google.com"

# load csv data
csv_to_read = "bg_ranks_ids_names.csv"
with open(csv_to_read, 'r', newline='', encoding='utf-8',errors='ignore') as games:
    read = csv.reader(games)
    # create and open new csv for writing
    with open('bg_prices.csv', 'w', newline='', encoding='utf-8') as new_csv:
        wr = csv.writer(new_csv)
        wr.writerow(["Rank","BGGid","Name","Price"])
        num_errors = 0
        mismatches = 0

        for row in read:
            rank, bgg_id, name = row[0], row[1], row[2]
            url = str(ENDPOINT+"&eid="+bgg_id)
            try:
                response = requests.get(url)
            except (ConnectionError, ConnectionRefusedError, ConnectionAbortedError, ConnectionResetError, RemoteDisconnected, ProtocolError, HTTPError):
                print("ERROR: Connection error for rank" + str(rank))
                num_errors += 1
                time.sleep(10)
            else:
                if response.status_code == 200:
                    data = response.json()
                    item_num = 0
                    new_row = ""
                    while (True):
                        try:
                            title = data['items'][item_num]['name']
                            price = data['items'][item_num]['prices'][0]['product']
                        except IndexError:
                            # use most recently found row of data if there was a name mismatch
                            #   and there are no more items or there is no price found
                            if (item_num >= len(data['items'])):
                                if (len(new_row) != 0):
                                    print(new_row) # optional, to keep an eye on processes while running
                                    wr.writerow(new_row)
                                    break
                                else:
                                    print("ERROR: Unable to find price for " + name)
                                    num_errors += 1
                                    break
                            elif (len(data['items']) == 0):
                                print("ERROR: No items returned on search for " + name)
                                num_errors += 1
                                break
                            elif (len(data['items'][item_num]['prices']) == 0):
                                # if no price found, go to next item
                                item_num += 1
                        else:
                            # check that names approximately match
                            # strip punctuation and whitespace, make all lowercase
                            clean_title = re.sub(r'[^a-zA-Z1-9]', '', title).lower()
                            clean_name = re.sub(r'[^a-zA-Z1-9]', '', name).lower()
                            if (clean_title not in clean_name) and (clean_name not in clean_title):
                                print(f"MISMATCH: {name} does not match {title}, ID = {bgg_id}")
                                # make row to add the first item and price returned
                                if (item_num == 0): new_row = [rank, bgg_id, name, price]
                                mismatches += 1
                                item_num += 1
                            else:
                                new_row = [rank, bgg_id, name, price]
                                print(new_row) # optional, to keep an eye on processes while running
                                wr.writerow(new_row)
                                break
                elif response.status_code == 429:
                    print("ERROR: Too many requests")
                    num_errors += 1
                    time.sleep(20)
                else: 
                    print("ERROR: Unsuccessful Request for " + url)
                    num_errors += 1

    print("entries not found: " + str(num_errors))
    print("mismatches: " + str(mismatches))