__author__ = "Aybuke Ozturk Suri, Johvany Gustave"
__copyright__ = "Copyright 2023, IN512, IPSA 2024"
__credits__ = ["Aybuke Ozturk Suri", "Johvany Gustave"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

from network import Network
from my_constants import *

from threading import Thread
import numpy as np
from time import sleep, time
import random


class Agent:
    def __init__(self, server_ip):
        # --- Navigation Memories ---
        self.visited_cells = set()
        self.previous_move = None
        self.last_value = 0.0

        # Stuck detection
        self.last_position = None
        self.stuck_counter = 0

        # --- Task State (Personnel) ---
        self.completed = False
        self.my_key_found = False  # True si J'AI ramassé ma clé
        self.my_box_location = None  # Position de ma boîte (si connue)
        self.target_pos = None  # Où je veux aller maintenant

        # --- Global Synchronization State (Collectif) ---
        self.agents_with_keys = set()  # Qui a ramassé sa clé ?
        self.agents_boxes_found = set()  # Pour qui a-t-on trouvé la boîte ?
        self.go_signal = False  # Le signal final

        # Memory of ALL found items
        self.discovered_items = {}  # Dict: (x,y) -> owner_id

        # Scanning Strategy
        self.scan_waypoints = []

        # Server msg container
        self.item_data_received = None

        # --- INIT NETWORK ---
        self.network = Network(server_ip=server_ip)
        self.agent_id = self.network.id
        self.running = True
        self.network.send({"header": GET_DATA})
        self.msg = {}
        env_conf = self.network.receive()
        self.nb_agent_expected = 0
        self.nb_agent_connected = 0
        self.x, self.y = env_conf["x"], env_conf["y"]
        self.w, self.h = env_conf["w"], env_conf["h"]

        Thread(target=self.msg_cb, daemon=True).start()
        self.wait_for_connected_agent()

    def msg_cb(self):
        """ Écoute des messages """
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
                # Enregistrement global des items découverts
                if "position" in msg and "owner" in msg:
                    self.discovered_items[msg["position"]] = msg["owner"]

                # Si une BOITE est découverte
                if msg.get("Msg type") == BOX_DISCOVERED:
                    owner = msg.get("owner")
                    self.agents_boxes_found.add(owner)
                    if owner == self.agent_id:
                        self.my_box_location = msg["position"]

                # Si une CLÉ est DÉCOUVERTE (vue mais pas forcement prise)
                elif msg.get("Msg type") == KEY_DISCOVERED:
                    owner = msg.get("owner")
                    # Si c'est ma clé, je la cible IMMEDIATEMENT
                    if owner == self.agent_id and not self.my_key_found:
                        self.target_pos = msg["position"]

                # Si une CLÉ est RAMASSÉE (Confirmation officielle)
                if msg.get("sub_type") == "KEY_COLLECTED":
                    sender = msg.get("sender")
                    self.agents_with_keys.add(sender)

    def wait_for_connected_agent(self):
        self.network.send({"header": GET_NB_AGENTS})
        check_conn_agent = True
        while check_conn_agent:
            if self.nb_agent_expected == self.nb_agent_connected:
                print("Tous les agents connectés !")
                check_conn_agent = False
            sleep(0.5)

    def get_real_val(self):
        return self.msg.get("cell_val", 0.0)

    def get_perceived_value(self):
        """
        Retourne la valeur de la case.
        Masque les objets déjà trouvés SAUF si c'est ma propre clé que je cherche.
        """
        real_val = self.get_real_val()

        for (ix, iy), owner in self.discovered_items.items():
            if abs(self.x - ix) <= 2 and abs(self.y - iy) <= 2:

                # IMPORTANT : Si c'est MA clé et que je ne l'ai pas, JE DOIS LA VOIR (ne pas masquer)
                if owner == self.agent_id and not self.my_key_found:
                    return real_val

                # Si c'est ma boite et que c'est le moment d'y aller
                if self.my_box_location == (ix, iy) and self.go_signal:
                    return real_val

                # Sinon (objet des autres ou ma boite trop tôt), je masque pour ne pas être distrait
                return 0.0

        return real_val

    def send_move(self, direction):
        self.network.send({"header": MOVE, "direction": direction})
        sleep(0.15)

    def check_item_owner(self):
        self.item_data_received = None
        self.network.send({"header": GET_ITEM_OWNER})
        # Petite sécurité anti-boucle infinie
        timeout = 0
        while self.item_data_received is None and timeout < 100:
            sleep(0.01)
            timeout += 1
        if self.item_data_received:
            return self.item_data_received.get("owner"), self.item_data_received.get("type")
        return None, None

    def check_global_conditions(self):
        """ Vérifie si TOUT a été trouvé """
        all_keys = len(self.agents_with_keys) >= self.nb_agent_expected
        all_boxes = len(self.agents_boxes_found) >= self.nb_agent_expected

        if all_keys and all_boxes:
            if not self.go_signal:
                print(f"Agent {self.agent_id}: TOUT EST TROUVÉ ! GO AUX BOITES !")
            self.go_signal = True
            self.target_pos = self.my_box_location

    def run_exploration(self):
        print(f"Agent {self.agent_id} démarre l'exploration...")

        moves_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1), 5: (-1, -1), 6: (1, -1), 7: (-1, 1), 8: (1, 1)}
        inverse_move = {1: 2, 2: 1, 3: 4, 4: 3, 5: 8, 6: 7, 7: 6, 8: 5}

        # Generation Zones de Scan
        if self.nb_agent_expected > 0:
            zone_width = self.w // self.nb_agent_expected
            min_x = self.agent_id * zone_width
            max_x = min_x + zone_width if self.agent_id != self.nb_agent_expected - 1 else self.w

            cx = min_x + 1
            down = True
            while cx < max_x:
                self.scan_waypoints.append((cx, 0) if down else (cx, self.h - 1))
                self.scan_waypoints.append((cx, self.h - 1) if down else (cx, 0))
                cx += 4
                down = not down

        while self.running and not self.completed:
            real_val = self.get_real_val()
            perceived_val = self.get_perceived_value()
            self.visited_cells.add((self.x, self.y))

            if self.last_position == (self.x, self.y):
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            self.last_position = (self.x, self.y)

            # --- ETAPE 0 : Verifier l'état global ---
            self.check_global_conditions()

            # --- ETAPE 1 : SUR UN ITEM (PRIORITÉ ABSOLUE) ---
            if real_val == 1.0:
                owner_id, item_type = self.check_item_owner()

                if owner_id is not None:
                    # 1. Broadcast la découverte
                    msg_type = KEY_DISCOVERED if item_type == KEY_TYPE else BOX_DISCOVERED
                    if (self.x, self.y) not in self.discovered_items:
                        self.network.send({
                            "header": BROADCAST_MSG,
                            "Msg type": msg_type,
                            "position": (self.x, self.y),
                            "owner": owner_id
                        })
                        self.discovered_items[(self.x, self.y)] = owner_id

                        if item_type == BOX_TYPE:
                            self.agents_boxes_found.add(owner_id)
                            if owner_id == self.agent_id:
                                self.my_box_location = (self.x, self.y)

                    # 2. LOGIQUE CRITIQUE DE RAMASSAGE
                    if owner_id == self.agent_id and item_type == KEY_TYPE:
                        if not self.my_key_found:
                            print(f"Agent {self.agent_id} a RAMASSÉ sa clé (Validation Locale).")
                            # UPDATE LOCAL IMMEDIAT (Ne pas attendre le réseau)
                            self.my_key_found = True
                            self.agents_with_keys.add(self.agent_id)
                            self.target_pos = None  # On a atteint la cible
                            self.last_value = 0

                            # Informer les autres
                            self.network.send({
                                "header": BROADCAST_MSG,
                                "sub_type": "KEY_COLLECTED",
                                "sender": self.agent_id
                            })

                            # Dégager de la case pour continuer
                            self.move_randomly_away(moves_map)
                            continue

                    # 3. Fin de partie (Sur ma boite + Signal)
                    if owner_id == self.agent_id and item_type == BOX_TYPE and self.go_signal:
                        self.completed = True
                        self.network.send({"header": BROADCAST_MSG, "Msg type": COMPLETED})
                        print("Mission Terminée !")
                        break

                # Si on est sur un item mais pas d'action spéciale, on s'éloigne
                if not self.target_pos:
                    self.move_randomly_away(moves_map)
                continue

            # --- ETAPE 2 : CIBLE DÉFINIE ---
            if self.target_pos:
                tx, ty = self.target_pos

                # Anti-blocage
                if self.stuck_counter > 3 and (self.x, self.y) != (tx, ty):
                    self.move_randomly_away(moves_map)
                    continue

                if self.x == tx and self.y == ty:
                    # On est arrivé, mais real_val n'était pas 1.0 au début de boucle ?
                    # Ça peut arriver avec la latence, on attend le prochain tour.
                    self.target_pos = None
                else:
                    move = self.get_move_towards(tx, ty, moves_map)
                    if move: self.send_move(move)
                    continue

            # --- ETAPE 3 : NAVIGATION ---

            # Obstacles
            if 0.34 <= real_val <= 0.36:
                if self.previous_move and self.previous_move in inverse_move:
                    self.send_move(inverse_move[self.previous_move])
                continue

            # Gradient Strict (Chasse à l'odeur)
            # Si on sent quelque chose, on ignore le scan et on cherche activement
            if perceived_val > 0:
                # Si la valeur a baissé, on a fait fausse route -> Demi-tour immédiat
                if perceived_val < self.last_value and self.previous_move in inverse_move:
                    self.send_move(inverse_move[self.previous_move])
                    # On "oublie" la last_value pour forcer un nouveau choix au prochain tour
                    self.last_value = 0
                    continue

                # Sinon (valeur monte ou stable), on continue ou on cherche meilleur voisin
                self.last_value = perceived_val

                valid = self.get_valid_moves(moves_map)
                # On essaie de privilégier les mouvements non visités
                best = [m for m in valid if
                        (self.x + moves_map[m][0], self.y + moves_map[m][1]) not in self.visited_cells]

                if best:
                    choice = random.choice(best)
                else:
                    choice = random.choice(valid)  # Repli si tout visité

                self.previous_move = choice
                self.send_move(choice)
                continue

            # Scan de Zone (Seulement si aucune odeur et aucune cible)
            if self.scan_waypoints:
                self.last_value = 0
                wp = self.scan_waypoints[0]
                if (self.x, self.y) == wp:
                    self.scan_waypoints.pop(0)
                    continue

                if self.stuck_counter > 3:
                    self.move_randomly_away(moves_map)
                    continue

                move = self.get_move_towards(wp[0], wp[1], moves_map)
                self.send_move(move)

            # Random Fallback
            else:
                self.last_value = 0
                self.move_randomly_away(moves_map)

    def move_randomly_away(self, moves_map):
        valid = self.get_valid_moves(moves_map)
        if valid:
            m = random.choice(valid)
            self.previous_move = m
            self.send_move(m)

    def get_valid_moves(self, moves_map):
        possible = []
        for m_id, (dx, dy) in moves_map.items():
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                possible.append(m_id)
        return possible

    def get_move_towards(self, tx, ty, moves_map):
        best_dist = float('inf')
        best_move = None
        for m in self.get_valid_moves(moves_map):
            nx, ny = self.x + moves_map[m][0], self.y + moves_map[m][1]
            dist = np.sqrt((nx - tx) ** 2 + (ny - ty) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_move = m
        return best_move


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="Ip address of the server", type=str, default="localhost")
    args = parser.parse_args()
    agent = Agent(args.server_ip)
    try:
        agent.run_exploration()
    except KeyboardInterrupt:
        pass