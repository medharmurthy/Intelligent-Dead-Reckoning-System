import math
import numpy as np

class PrototypeMapMatcher:
    """
    Prototype Map Matching engine for road-network constraint (Sections 51-52 of plan.pdf).
    Score S = w_d * D + w_h * H + w_t * T
      - D: Distance from road candidate
      - H: Heading mismatch
      - T: Topology mismatch
    """
    def __init__(self, w_d=0.5, w_h=0.3, w_t=0.2):
        self.w_d = w_d
        self.w_h = w_h
        self.w_t = w_t
        self.roads = [] # List of road line segments [(x1, y1, x2, y2, road_heading, road_id)]

    def load_geojson_roads(self, road_segments):
        """
        Loads road segments.
        Each segment is dict: {'id': id, 'p1': (x1, y1), 'p2': (x2, y2), 'heading': deg}
        """
        self.roads = road_segments

    def match_position(self, est_x, est_y, est_heading_deg, search_radius=50.0):
        """
        Matches position against candidate road segments within search_radius.
        Returns:
          matched_x, matched_y, matched_heading_deg, map_confidence
        """
        if not self.roads:
            # If no map data loaded, return unconstrained position with default confidence
            return est_x, est_y, est_heading_deg, 0.5
            
        best_score = float('inf')
        best_candidate = None
        
        for road in self.roads:
            x1, y1 = road['p1']
            x2, y2 = road['p2']
            road_hdg = road.get('heading', 0.0)
            
            # Point to segment distance D
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx*dx + dy*dy
            
            if seg_len_sq == 0:
                proj_x, proj_y = x1, y1
                dist = math.hypot(est_x - x1, est_y - y1)
            else:
                t = max(0.0, min(1.0, ((est_x - x1)*dx + (est_y - y1)*dy) / seg_len_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = math.hypot(est_x - proj_x, est_y - proj_y)
                
            if dist > search_radius:
                continue
                
            # Heading mismatch H (degrees normalized to [0, 1])
            hdg_diff = abs((est_heading_deg - road_hdg + 180) % 360 - 180)
            H = hdg_diff / 180.0
            
            # Normalized Distance D
            D = dist / search_radius
            
            # Topology penalty T (0 if close and aligned)
            T = 0.0 if dist < 10.0 else 0.5
            
            score = self.w_d * D + self.w_h * H + self.w_t * T
            
            if score < best_score:
                best_score = score
                best_candidate = (proj_x, proj_y, road_hdg)
                
        if best_candidate is not None:
            proj_x, proj_y, road_hdg = best_candidate
            map_confidence = max(0.0, min(1.0, 1.0 - best_score))
            return proj_x, proj_y, road_hdg, map_confidence
        else:
            return est_x, est_y, est_heading_deg, 0.1
