__author__ = "Aybuke Ozturk Suri, Johvany Gustave"
__copyright__ = "Copyright 2023, IN512, IPSA 2024"
__credits__ = ["Aybuke Ozturk Suri", "Johvany Gustave"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

from network import Network
from my_constants import *

from threading import Thread
import numpy as np
from time import sleep
import random


class Agent:
    """ Class that implements the behaviour of each agent based on their perception and communication with other agents """

    def __init__(self, server_ip):
        # Navigation memories
        self.visited_cells = set()
        self.previous_move = None
        self.last_value = 0.0

        # Task state
        self.completed = False
        self.my_key_found = False
        self.my_box_location = None
        self.target_pos = None

        # Synchronization State (NEW)
        self.am_i_ready = False
        self.ready_agents = set()  # Set of Agent IDs who are ready to finish
        self.go_signal = False  # Becomes True when everyone is ready

        # Memory of ALL found items to create "Virtual 0" zones
        self.discovered_items = set()

        # Container for server response
        self.item_data_received = None

        # DO NOT TOUCH THE FOLLOWING INSTRUCTIONS
        self.network = Network(server_ip=server_ip)
        self.agent_id = self.network.id
        self.running = True
        self.network.send({"header": GET_DATA})
        self.msg = {}
        env_conf = self.network.receive()
        self.nb_agent_expected = 0
        self.nb_agent_connected = 0
        self.x, self.y = env_conf["x"], env_conf["y"]  # initial agent position
        self.w, self.h = env_conf["w"], env_conf["h"]  # environment dimensions
        cell_val = env_conf["cell_val"]  # value of the cell the agent is located in
        print(cell_val)
        Thread(target=self.msg_cb, daemon=True).start()
        print("hello")
        self.wait_for_connected_agent()

    def msg_cb(self):
        """ Method used to handle incoming messages """
        while self.running:
            msg = self.network.receive()
            self.msg = msg

            if msg["header"] == MOVE:
                self.x, self.y = msg["x"], msg["y"]

            elif msg["header"] == GET_NB_AGENTS:
                self.nb_agent_expected = msg["nb_agents"]

            elif msg["header"] == GET_NB_CONNECTED_AGENTS:
                self.nb_agent_connected = msg["nb_connected_agents"]

            elif msg["header"] == GET_ITEM_OWNER:
                self.item_data_received = msg

            elif msg["header"] == BROADCAST_MSG:
                # 1. Store location to apply "Virtual 0" mask
                if "position" in msg:
                    self.discovered_items.add(msg["position"])

                # 2. Synchronization Check (NEW)
                # We check for a custom "sub_type" in the broadcast dictionary
                if "sub_type" in msg and msg["sub_type"] == "READY":
                    sender_id = msg.get("sender") if "sender" in msg else msg.get("owner")  # Safety check
                    print(f"Agent {self.agent_id}: Received READY signal from Agent {sender_id}")
                    self.ready_agents.add(sender_id)

                # 3. Item Ownership logic
                if "owner" in msg and msg["owner"] == self.agent_id:
                    if msg["Msg type"] == KEY_DISCOVERED and not self.my_key_found:
                        self.target_pos = msg["position"]
                    elif msg["Msg type"] == BOX_DISCOVERED:
                        self.my_box_location = msg["position"]

    def wait_for_connected_agent(self):
        self.network.send({"header": GET_NB_AGENTS})
        check_conn_agent = True
        while check_conn_agent:
            if self.nb_agent_expected == self.nb_agent_connected:
                print("both connected!")
                check_conn_agent = False
            sleep(0.5)

            # TODO: CREATE YOUR METHODS HERE...

    def get_real_val(self):
        return self.msg.get("cell_val", 0.0)

    def get_perceived_value(self):
        """
        Returns value, masking known items to 0.0.
        Critical for Synchronization: If I am ready but waiting for others,
        I must mask MY OWN BOX (return 0.0) so I don't sit on it.
        """
        real_val = self.get_real_val()

        for (ix, iy) in self.discovered_items:
            if abs(self.x - ix) <= 2 and abs(self.y - iy) <= 2:

                # If I have a target (like my key), I see it.
                if self.target_pos and (ix, iy) == self.target_pos:
                    return real_val

                # NEW: If the GO SIGNAL is active, I see my box.
                if self.go_signal and self.my_box_location == (ix, iy):
                    return real_val

                return 0.0

        return real_val

    def send_move(self, direction):
        self.network.send({"header": MOVE, "direction": direction})
        sleep(0.15)

    def check_item_owner(self):
        self.item_data_received = None
        self.network.send({"header": GET_ITEM_OWNER})
        while self.item_data_received is None:
            sleep(0.01)
        return self.item_data_received.get("owner"), self.item_data_received.get("type")

    def check_global_sync(self):
        """ Checks if all agents are ready to finish """
        # If I am ready, ensure I am in my own set
        if self.am_i_ready:
            self.ready_agents.add(self.agent_id)

        # Check if everyone is ready
        if len(self.ready_agents) >= self.nb_agent_expected:
            if not self.go_signal:
                print(f"Agent {self.agent_id}: ALL AGENTS READY! GOING TO BOX.")
            self.go_signal = True
            self.target_pos = self.my_box_location

    def run_exploration(self):
        print(f"Agent {self.agent_id} starting exploration...")

        moves_map = {
            1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1),
            5: (-1, -1), 6: (1, -1), 7: (-1, 1), 8: (1, 1)
        }
        inverse_move = {
            1: 2, 2: 1, 3: 4, 4: 3, 5: 8, 6: 7, 7: 6, 8: 5
        }
        self.previous_move = None

        while self.running and not self.completed:
            real_val = self.get_real_val()
            perceived_val = self.get_perceived_value()
            self.visited_cells.add((self.x, self.y))

            # --- STEP 0: CHECK SYNCHRONIZATION ---
            # Check if I just became ready (Have Key + Know Box Location)
            if not self.am_i_ready and self.my_key_found and self.my_box_location:
                self.am_i_ready = True
                self.ready_agents.add(self.agent_id)
                print(f"Agent {self.agent_id} is READY. Waiting for teammates...")
                # Broadcast READY status
                self.network.send({
                    "header": BROADCAST_MSG,
                    "sub_type": "READY",  # Custom field
                    "sender": self.agent_id
                })

            # Check if everyone else is ready
            self.check_global_sync()

            # --- CASE 1: MOVE TOWARDS TARGET ---
            if self.target_pos:
                target_x, target_y = self.target_pos

                if self.x == target_x and self.y == target_y:

                    # Logic for finishing
                    if self.my_key_found and self.my_box_location == (self.x, self.y):
                        # Only finish if global sync is true
                        if self.go_signal:
                            self.completed = True
                            self.network.send({"header": BROADCAST_MSG, "Msg type": COMPLETED})
                            print("Task Completed!")
                            break
                        else:
                            # Reached box but teammates aren't ready.
                            # Reset target and wander (perceived_val will mask box to 0)
                            print("At box, but waiting for team. Exploring...")
                            self.target_pos = None
                    else:
                        # Reached a target that wasn't the final box (e.g. Key)
                        self.target_pos = None
                else:
                    best_dir = self.get_move_towards(target_x, target_y, moves_map)
                    if best_dir:
                        self.send_move(best_dir)
                        continue

            # --- CASE 2: FOUND ITEM ---
            if real_val == 1.0:
                if (self.x, self.y) not in self.discovered_items:
                    owner_id, item_type = self.check_item_owner()
                    if owner_id is not None:
                        msg_type = KEY_DISCOVERED if item_type == KEY_TYPE else BOX_DISCOVERED

                        print(f"Agent {self.agent_id} discovered {item_type} for Agent {owner_id}")
                        self.network.send({
                            "header": BROADCAST_MSG,
                            "Msg type": msg_type,
                            "position": (self.x, self.y),
                            "owner": owner_id
                        })
                        self.discovered_items.add((self.x, self.y))

                        if owner_id == self.agent_id:
                            if item_type == KEY_TYPE:
                                if not self.my_key_found:
                                    self.my_key_found = True
                                    print("Collected MY KEY.")
                                    self.last_value = 0
                                    # Note: We don't go to box yet, we wait for the Sync Check at top of loop
                                    self.move_randomly_away(moves_map)
                            elif item_type == BOX_TYPE:
                                self.my_box_location = (self.x, self.y)
                                self.move_randomly_away(moves_map)
                        else:
                            self.move_randomly_away(moves_map)
                    else:
                        self.move_randomly_away(moves_map)
                else:
                    self.move_randomly_away(moves_map)
                continue

            # --- CASE 3: OBSTACLE AVOIDANCE ---
            if 0.34 <= real_val <= 0.36:
                if self.previous_move and self.previous_move in inverse_move:
                    self.send_move(inverse_move[self.previous_move])
                continue

            # --- CASE 4: GRADIENT ASCENT ---
            if perceived_val > 0:
                if perceived_val < self.last_value:
                    if self.previous_move and self.previous_move in inverse_move:
                        self.send_move(inverse_move[self.previous_move])
                        self.last_value = perceived_val
                        continue

                valid_moves = self.get_possible_moves(moves_map)
                unvisited_moves = [m for m in valid_moves if
                                   (self.x + moves_map[m][0], self.y + moves_map[m][1]) not in self.visited_cells]
                move_choice = random.choice(unvisited_moves) if unvisited_moves else random.choice(valid_moves)
                self.previous_move = move_choice
                self.last_value = perceived_val
                self.send_move(move_choice)

            # --- CASE 5: RANDOM EXPLORATION ---
            else:
                self.last_value = 0
                valid_moves = self.get_possible_moves(moves_map)
                unvisited_moves = [m for m in valid_moves if
                                   (self.x + moves_map[m][0], self.y + moves_map[m][1]) not in self.visited_cells]

                if unvisited_moves:
                    move_choice = random.choice(unvisited_moves)
                else:
                    move_choice = random.choice(valid_moves)

                self.previous_move = move_choice
                self.send_move(move_choice)

    def move_randomly_away(self, moves_map):
        valid = self.get_possible_moves(moves_map)
        if valid:
            m = random.choice(valid)
            self.previous_move = m
            self.send_move(m)

    def get_possible_moves(self, moves_map):
        possible = []
        for m_id, (dx, dy) in moves_map.items():
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                possible.append(m_id)
        return possible

    def get_move_towards(self, tx, ty, moves_map):
        best_dist = float('inf')
        best_move = None
        valid_moves = self.get_possible_moves(moves_map)
        for m in valid_moves:
            nx = self.x + moves_map[m][0]
            ny = self.y + moves_map[m][1]
            dist = np.sqrt((nx - tx) ** 2 + (ny - ty) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_move = m
        return best_move


if __name__ == "__main__":
    from random import randint
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="Ip address of the server", type=str, default="localhost")
    args = parser.parse_args()

    agent = Agent(args.server_ip)

    try:
        agent.run_exploration()
    except KeyboardInterrupt:
        pass