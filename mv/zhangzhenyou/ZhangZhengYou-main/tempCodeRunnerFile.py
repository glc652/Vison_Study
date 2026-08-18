# 对上下端角点进行额外的边界精化
            if i in [0, 1]:  # 上端角点
                refined = find_edge_boundary(edges, refined, direction='top', search_range=30)
            elif i in [2, 3]:  # 下端角点
                refined = find_edge_boundary(edges, refined, direction='bottom', search_range=15)
