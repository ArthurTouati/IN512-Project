__author__ = "Aybuke Ozturk Suri, Johvany Gustave"
__copyright__ = "Copyright 2023, IN512, IPSA 2024"
__credits__ = ["Aybuke Ozturk Suri", "Johvany Gustave"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

from network import Network
from my_constants import *

from threading import Thread, Lock
import numpy as np
from time import sleep
from collections import deque
import sys


class Agent:
    """ Multi-agent exploration with aggressive wall avoidance """
    def __init__(self, server_ip):
        # Exploration state
        self.visited = set()
        self.exploration_map = {}
        self.current_cell_val = 0
        self.blocked_cells = set()
        self.item_cells = set()
        
        # Previous position for backtracking
        self.prev_x, self.prev_y = 0, 0
        self.previous_move = None
        self.last_value = 0.0  # For gradient tracking
        
        # Item discovery
        self.my_key_collected = False
        self.my_key_location = None
        self.my_box_location = None
        self.all_keys = {}
        self.all_boxes = {}
        
        # Coordination
        self.completed = False
        
        # Lane exploration
        self.my_zone_start = 0
        self.my_zone_end = 0
        self.lane_spacing = 4
        self.current_lane = 0
        self.lane_direction = 1
        
        # Thread safety
        self.lock = Lock()
        
        # Directions
        self.directions = {
            STAND: (0, 0),
            LEFT: (-1, 0),
            RIGHT: (1, 0),
            UP: (0, -1),
            DOWN: (0, 1),
            UP_LEFT: (-1, -1),
            UP_RIGHT: (1, -1),
            DOWN_LEFT: (-1, 1),
            DOWN_RIGHT: (1, 1)
        }
        
        # Inverse directions for gradient backtracking
        self.inverse_directions = {
            LEFT: RIGHT,
            RIGHT: LEFT,
            UP: DOWN,
            DOWN: UP,
            UP_LEFT: DOWN_RIGHT,
            UP_RIGHT: DOWN_LEFT,
            DOWN_LEFT: UP_RIGHT,
            DOWN_RIGHT: UP_LEFT
        }
        
        # Network setup
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
        cell_val = env_conf["cell_val"]
        
        self.prev_x, self.prev_y = self.x, self.y
        self.visited.add((self.x, self.y))
        self.exploration_map[(self.x, self.y)] = cell_val
        self.current_cell_val = cell_val
        
        print(f"Agent {self.agent_id} at ({self.x}, {self.y})")
        
        Thread(target=self.msg_cb, daemon=True).start()
        self.wait_for_connected_agent()
        self.setup_zones()


    def block_wall_zone(self, x, y):
        """
        When we detect a wall cell (0.35 or confirmed wall),
        block just that cell to prevent entering it.
        """
        pos = (x, y)
        if pos not in self.item_cells:
            self.blocked_cells.add(pos)


    def on_wall_aura(self, x, y, cell_val):
        """
        Called when we step on a 0.35 cell (wall aura).
        Block just this cell and predict where the wall is ahead.
        """
        # Block just this cell (the aura)
        if (x, y) not in self.item_cells:
            self.blocked_cells.add((x, y))
        
        # Predict where the wall is based on previous move direction
        # Only block cells AHEAD (where the wall likely is)
        if self.previous_move and self.previous_move in self.directions:
            dx, dy = self.directions[self.previous_move]
            if dx != 0 or dy != 0:  # Not STAND
                # The wall is likely 1 cell ahead in the direction we were moving
                wall_x, wall_y = x + dx, y + dy
                if 0 <= wall_x < self.w and 0 <= wall_y < self.h:
                    if (wall_x, wall_y) not in self.item_cells:
                        self.blocked_cells.add((wall_x, wall_y))


    def msg_cb(self):
        while self.running:
            try:
                msg = self.network.receive()
                if msg is None:
                    continue
                    
                with self.lock:
                    self.msg = msg
                    header = msg.get("header")
                    
                    if header == MOVE:
                        # Save previous position
                        self.prev_x, self.prev_y = self.x, self.y
                        
                        self.x, self.y = msg["x"], msg["y"]
                        self.current_cell_val = msg.get("cell_val", 0)
                        pos = (self.x, self.y)
                        self.visited.add(pos)
                        self.exploration_map[pos] = self.current_cell_val
                        
                        # Check for wall aura
                        if abs(self.current_cell_val - 0.35) < 0.02:
                            self.on_wall_aura(self.x, self.y, self.current_cell_val)
                        
                    elif header == GET_NB_AGENTS:
                        self.nb_agent_expected = msg["nb_agents"]
                        
                    elif header == GET_NB_CONNECTED_AGENTS:
                        self.nb_agent_connected = msg["nb_connected_agents"]
                        
                    elif header == GET_ITEM_OWNER:
                        self.handle_item_owner(msg)
                        
                    elif header == BROADCAST_MSG:
                        self.handle_broadcast(msg)
                        
            except Exception as e:
                pass


    def wait_for_connected_agent(self):
        self.network.send({"header": GET_NB_AGENTS})
        while True:
            sleep(0.1)
            self.network.send({"header": GET_NB_CONNECTED_AGENTS})
            sleep(0.1)
            with self.lock:
                if self.nb_agent_expected > 0 and self.nb_agent_expected == self.nb_agent_connected:
                    print(f"Agent {self.agent_id}: All connected!")
                    break
    
    
    def setup_zones(self):
        zone_height = self.h // self.nb_agent_expected
        self.my_zone_start = self.agent_id * zone_height
        self.my_zone_end = (self.agent_id + 1) * zone_height - 1
        if self.agent_id == self.nb_agent_expected - 1:
            self.my_zone_end = self.h - 1
        self.current_lane = self.my_zone_start
        self.lane_direction = 1 if self.x < self.w // 2 else -1
        print(f"Agent {self.agent_id}: Zone {self.my_zone_start}-{self.my_zone_end}")
    
    
    def is_blocked(self, pos):
        x, y = pos
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return True
        
        if pos in self.item_cells:
            return False
        
        if pos in self.blocked_cells:
            return True
        
        cell_val = self.exploration_map.get(pos, None)
        if cell_val is not None and abs(cell_val - 0.35) < 0.02:
            self.block_wall_zone(pos[0], pos[1])
            return True
        
        return False
    
    
    def is_on_item_aura(self):
        """
        Check if we're on an item aura (not wall aura).
        Key aura: 0.25 -> 0.5 -> 1.0
        Box aura: 0.3 -> 0.6 -> 1.0
        Wall aura: 0.35
        Returns True if on key or box aura (but not on the item itself at 1.0)
        Only returns True if the item hasn't been discovered yet.
        """
        val = self.current_cell_val
        
        # Exclude wall aura (0.35) and empty cells (0)
        if val < 0.2 or abs(val - 0.35) < 0.03:
            return False
        
        # Exclude the item itself (1.0)
        if abs(val - 1.0) < 0.01:
            return False
        
        # Key aura range: 0.25 to 0.5
        # Box aura range: 0.3 to 0.6
        # Combined: anything from 0.2 to 0.7 (excluding wall at 0.35)
        if 0.2 <= val <= 0.7:
            # Check if we're near an already-discovered item
            # If so, don't follow this gradient anymore
            current_pos = (self.x, self.y)
            
            # Check all known item positions
            all_known_items = list(self.all_keys.values()) + list(self.all_boxes.values())
            for item_pos in all_known_items:
                # If within 3 cells of a known item, this is its aura - skip it
                dist = abs(current_pos[0] - item_pos[0]) + abs(current_pos[1] - item_pos[1])
                if dist <= 3:
                    return False
            
            return True
        
        return False
    
    
    def follow_item_gradient(self):
        """
        When on an item aura, move toward adjacent cell with highest value
        to find the item (which has value 1.0).
        Prioritizes: 1) Known higher values, 2) Unvisited adjacent cells
        """
        best_pos = None
        best_val = self.current_cell_val
        unvisited_candidates = []
        
        # Check all 8 adjacent cells
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), (-1,1), (1,1)]:
            nx, ny = self.x + dx, self.y + dy
            
            # Skip out of bounds
            if nx < 0 or nx >= self.w or ny < 0 or ny >= self.h:
                continue
            
            # Skip blocked cells (but not item cells)
            pos = (nx, ny)
            if pos in self.blocked_cells and pos not in self.item_cells:
                continue
            
            # Get cell value
            cell_val = self.exploration_map.get(pos, None)
            
            if cell_val is not None:
                # Skip wall aura cells
                if abs(cell_val - 0.35) < 0.03:
                    continue
                # Found a higher value - potential path to item
                if cell_val > best_val:
                    best_val = cell_val
                    best_pos = pos
            else:
                # Unvisited cell - we should explore it to find the item
                unvisited_candidates.append(pos)
        
        # Priority 1: Move to known higher value (closer to item)
        if best_pos:
            return self.direction_to(best_pos)
        
        # Priority 2: Explore unvisited adjacent cells
        if unvisited_candidates:
            # Pick the first unvisited cell to explore
            return self.direction_to(unvisited_candidates[0])
        
        # No better adjacent cell found, continue normal exploration
        return None
    
    
    def direction_to(self, target):
        dx = target[0] - self.x
        dy = target[1] - self.y
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        for d, (ddx, ddy) in self.directions.items():
            if ddx == dx and ddy == dy:
                return d
        return STAND
    
    def get_perceived_value(self):
        """
        Returns the cell value but masks already-discovered items.
        Exception: Shows own key if not yet collected, and own box if go signal is active.
        """
        val = self.current_cell_val
        current_pos = (self.x, self.y)
        
        # Check all known item positions
        all_known_items = {}
        for owner, pos in self.all_keys.items():
            all_known_items[pos] = ('key', owner)
        for owner, pos in self.all_boxes.items():
            all_known_items[pos] = ('box', owner)
        
        for item_pos, (item_type, owner) in all_known_items.items():
            # Check if we're in range of this item's aura (within 2 cells)
            if abs(self.x - item_pos[0]) <= 2 and abs(self.y - item_pos[1]) <= 2:
                # Don't mask my own key that I haven't collected yet
                if item_type == 'key' and owner == self.agent_id and not self.my_key_collected:
                    return val
                # Don't mask my own box if ready to go
                if item_type == 'box' and owner == self.agent_id and self.ready_for_box():
                    return val
                # Mask other items to avoid distraction
                return 0.0
        
        return val
    
    
    def get_valid_exploration_moves(self):
        """Get valid moves that don't lead to walls, blocked cells, or out of bounds"""
        valid_moves = []
        for d, (dx, dy) in self.directions.items():
            if d == STAND:
                continue
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                if not self.is_blocked((nx, ny)):
                    # Also check if this cell has wall aura value in exploration_map
                    cell_val = self.exploration_map.get((nx, ny), None)
                    if cell_val is not None and abs(cell_val - 0.35) < 0.03:
                        # This is a wall aura cell - block it
                        self.block_wall_zone(nx, ny)
                        continue
                    
                    # For diagonal moves, check that we won't clip through walls
                    if abs(dx) == 1 and abs(dy) == 1:
                        # Check both cardinal components
                        side1 = (self.x + dx, self.y)
                        side2 = (self.x, self.y + dy)
                        if self.is_blocked(side1) or self.is_blocked(side2):
                            continue
                    
                    valid_moves.append(d)
        return valid_moves
    
    
    def explore_lane(self):
        import random
        
        # Check if currently on wall aura or in blocked cell - escape!
        current_pos = (self.x, self.y)
        if abs(self.current_cell_val - 0.35) < 0.02 or current_pos in self.blocked_cells:
            self.last_value = 0
            return self.escape_from_trap()
        
        # Get perceived value (masks already-discovered items)
        perceived_val = self.get_perceived_value()
        
        # GRADIENT FOLLOWING: If we sense something (item aura), chase it!
        if perceived_val > 0 and abs(perceived_val - 0.35) > 0.03:  # Not wall aura
            # Get valid moves first
            valid_moves = self.get_valid_exploration_moves()
            
            # If value dropped, we went the wrong way - turn back immediately!
            if perceived_val < self.last_value and self.previous_move in self.inverse_directions:
                inverse_dir = self.inverse_directions[self.previous_move]
                # Only use inverse direction if it's valid (not blocked)
                if inverse_dir in valid_moves:
                    self.last_value = 0  # Reset to force new choice next turn
                    self.previous_move = inverse_dir
                    return inverse_dir
                # If inverse is blocked, fall through to pick from valid_moves
            
            # Value is increasing or stable - continue exploring
            self.last_value = perceived_val
            
            if valid_moves:
                # Prefer unvisited cells
                unvisited = [m for m in valid_moves 
                            if (self.x + self.directions[m][0], 
                                self.y + self.directions[m][1]) not in self.visited]
                
                if unvisited:
                    choice = random.choice(unvisited)
                else:
                    choice = random.choice(valid_moves)
                
                self.previous_move = choice
                return choice
        
        # No gradient detected - reset last_value and do normal lane exploration
        self.last_value = 0
        
        if self.y != self.current_lane:
            d = self.navigate_to((self.x, self.current_lane))
            if d != STAND:
                self.previous_move = d
                return d
            self.current_lane += self.lane_spacing
            if self.current_lane > self.my_zone_end:
                self.current_lane = self.my_zone_start + 2
        
        next_x = self.x + self.lane_direction
        
        if next_x < 0 or next_x >= self.w or self.is_blocked((next_x, self.y)):
            self.lane_direction *= -1
            self.current_lane += self.lane_spacing
            if self.current_lane > self.my_zone_end:
                self.current_lane = self.my_zone_start + 2
                if self.current_lane > self.my_zone_end:
                    self.current_lane = self.my_zone_start
            d = self.navigate_to((self.x, self.current_lane))
            self.previous_move = d
            return d
        
        d = self.direction_to((next_x, self.y))
        self.previous_move = d
        return d
    
    
    def escape_from_trap(self):
        """
        Emergency escape when stuck inside blocked area (like L-shaped obstacle).
        Use BFS to find the nearest cell that is NOT in blocked_cells and NOT a wall.
        Only considers cardinal directions to avoid diagonal clipping through walls.
        """
        queue = deque([(self.x, self.y, [])])
        seen = {(self.x, self.y)}
        
        while queue:
            cx, cy, path = queue.popleft()
            
            # Only use cardinal directions for escape to avoid clipping
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    
                    # Check if this cell is a potential safe escape destination
                    cell_val = self.exploration_map.get((nx, ny), None)
                    is_wall = cell_val is not None and (abs(cell_val - 0.35) < 0.02 or abs(cell_val - 1.0) < 0.01)
                    
                    # If not blocked and not a wall, this is our escape route
                    if (nx, ny) not in self.blocked_cells and not is_wall:
                        full_path = path + [(nx, ny)]
                        if full_path:
                            return self.direction_to(full_path[0])
                    
                    # Continue searching even through blocked cells to find exit
                    queue.append((nx, ny, path + [(nx, ny)]))
        
        # No escape found - try any cardinal direction that's in bounds
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                return self.direction_to((nx, ny))
        
        return STAND
    
    
    def navigate_to(self, target):
        if not target or (self.x, self.y) == target:
            return STAND
        
        # First, try to find a path using only KNOWN (visited) cells
        queue = deque([(self.x, self.y, [])])
        seen = {(self.x, self.y)}
        
        while queue:
            cx, cy, path = queue.popleft()
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), (-1,1), (1,1)]:
                nx, ny = cx + dx, cy + dy
                
                if not (0 <= nx < self.w and 0 <= ny < self.h):
                    continue
                
                if (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                
                # Check if we reached the target
                if (nx, ny) == target:
                    # Target is allowed even if not in exploration_map (it's an item)
                    full_path = path + [(nx, ny)]
                    if full_path:
                        return self.direction_to(full_path[0])
                    return STAND
                
                # Check if cell is blocked
                if self.is_blocked((nx, ny)):
                    continue
                
                # Check cell value for wall aura
                cell_val = self.exploration_map.get((nx, ny), None)
                
                # Skip cells we haven't visited yet (unknown territory)
                # This prevents pathing through walls we haven't discovered
                if cell_val is None:
                    continue
                
                if abs(cell_val - 0.35) < 0.03:
                    self.blocked_cells.add((nx, ny))
                    continue
                
                # For diagonal moves, check that we can pass through
                if abs(dx) == 1 and abs(dy) == 1:
                    side1 = (cx + dx, cy)
                    side2 = (cx, cy + dy)
                    if self.is_blocked(side1) or self.is_blocked(side2):
                        continue
                    side1_val = self.exploration_map.get(side1, None)
                    side2_val = self.exploration_map.get(side2, None)
                    if side1_val is None or side2_val is None:
                        continue
                    if abs(side1_val - 0.35) < 0.03 or abs(side2_val - 0.35) < 0.03:
                        continue
                
                queue.append((nx, ny, path + [(nx, ny)]))
        
        # No known path found - move toward target by exploring
        # Find best valid move that gets us closer to target
        valid_moves = self.get_valid_exploration_moves()
        if valid_moves:
            target_x, target_y = target
            best_move = None
            best_dist = float('inf')
            for move in valid_moves:
                dx, dy = self.directions[move]
                nx, ny = self.x + dx, self.y + dy
                dist = abs(nx - target_x) + abs(ny - target_y)
                if dist < best_dist:
                    best_dist = dist
                    best_move = move
            if best_move:
                return best_move
        
        # Truly stuck - try escape
        return self.escape_from_trap()
    
    
    def check_cell(self):
        with self.lock:
            cell_val = self.current_cell_val
        if abs(cell_val - 1.0) < 0.01:
            self.network.send({"header": GET_ITEM_OWNER})
            sleep(0.15)
    
    
    def handle_item_owner(self, msg):
        owner = msg.get("owner")
        item_type = msg.get("type")
        pos = (self.x, self.y)
        
        if owner is None:
            # This is a WALL, not an item - block just this cell
            pos = (self.x, self.y)
            if pos not in self.item_cells:
                self.blocked_cells.add(pos)
            return
        
        self.item_cells.add(pos)
        self.blocked_cells.discard(pos)
        
        if item_type == KEY_TYPE:
            self.all_keys[owner] = pos
            if owner == self.agent_id:
                self.my_key_collected = True
                self.my_key_location = pos
                print(f"Agent {self.agent_id}: *** GOT MY KEY ***")
            self.broadcast_item(owner, item_type, pos)
                
        elif item_type == BOX_TYPE:
            self.all_boxes[owner] = pos
            if owner == self.agent_id:
                self.my_box_location = pos
                print(f"Agent {self.agent_id}: *** FOUND MY BOX ***")
            self.broadcast_item(owner, item_type, pos)
    
    
    def broadcast_item(self, owner, item_type, pos):
        self.network.send({
            "header": BROADCAST_MSG,
            "Msg type": KEY_DISCOVERED if item_type == KEY_TYPE else BOX_DISCOVERED,
            "position": pos,
            "owner": owner
        })
    
    
    def handle_broadcast(self, msg):
        if msg is None:
            return
        msg_type = msg.get("Msg type")
        pos_data = msg.get("position")
        if pos_data is None:
            return
        pos = tuple(pos_data)
        owner = msg.get("owner")
        
        self.item_cells.add(pos)
        self.blocked_cells.discard(pos)
        
        if msg_type == KEY_DISCOVERED:
            self.all_keys[owner] = pos
            if owner == self.agent_id and not self.my_key_location:
                self.my_key_location = pos
                print(f"Agent {self.agent_id}: Key location: {pos}")
                
        elif msg_type == BOX_DISCOVERED:
            self.all_boxes[owner] = pos
            if owner == self.agent_id and not self.my_box_location:
                self.my_box_location = pos
                print(f"Agent {self.agent_id}: Box location: {pos}")
                
        elif msg_type == COMPLETED:
            pass
    
    
    def ready_for_box(self):
        return (self.my_key_collected and 
                self.my_box_location and
                len(self.all_keys) >= self.nb_agent_expected and 
                len(self.all_boxes) >= self.nb_agent_expected)
    
    
    def need_key(self):
        return self.my_key_location and not self.my_key_collected
    
    
    def run(self):
        print(f"Agent {self.agent_id}: Starting...")
        
        while not self.completed and self.running:
            try:
                self.check_cell()
                
                with self.lock:
                    pos = (self.x, self.y)
                
                if self.need_key():
                    if pos == self.my_key_location:
                        self.check_cell()
                        sleep(0.2)
                    else:
                        d = self.navigate_to(self.my_key_location)
                        self.network.send({"header": MOVE, "direction": d})
                
                elif self.ready_for_box():
                    if pos == self.my_box_location:
                        self.completed = True
                        self.network.send({
                            "header": BROADCAST_MSG,
                            "Msg type": COMPLETED,
                            "position": pos,
                            "owner": self.agent_id
                        })
                        print(f"Agent {self.agent_id}: *** MISSION COMPLETE! *** Visited: {len(self.visited)}")
                        break
                    else:
                        d = self.navigate_to(self.my_box_location)
                        self.network.send({"header": MOVE, "direction": d})
                
                else:
                    d = self.explore_lane()
                    self.network.send({"header": MOVE, "direction": d})
                
                sleep(0.1)
                
            except Exception as e:
                sleep(0.1)
        
        print(f"Agent {self.agent_id}: Done!")
        self.running = False
        sys.exit(0)

             
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="Server IP", type=str, default="localhost")
    args = parser.parse_args()

    agent = Agent(args.server_ip)
    
    try:
        agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        agent.running = False
        sys.exit(0)