from math import sqrt


def normalize(vector):
    """Scale the vector to a length of 1.

    Args:
        vector (list): The original vector.

    Returns:
        list: The normalized vector.
    """
    norm = sqrt(vector[0] ** 2 + vector[1] ** 2)
    return vector[0] / norm, vector[1] / norm


def dot(vector1, vector2):
    """Calculate the dot product of two vectors.

    Args:
        vector1 (list): The first vector.
        vector2 (list): The second vector.
    
    Returns:
        float: The dot (or scalar) product of the two vectors.
    """
    return vector1[0] * vector2[0] + vector1[1] * vector2[1]


def edge_direction(point0, point1):
    """Calculate the direction of an edge.

    Args:
        point0 (list): The first point.
        point1 (list): The second point.

    Returns:
        list: A vector going from point0 to point1.
    """
    return point1[0] - point0[0], point1[1] - point0[1]


def orthogonal(vector):
    """Calculate the orthogonal vector of a given vector.

    Args:
        vector (list): The original vector.

    Returns:
        list: A new vector which is orthogonal to the given vector.
    """
    return vector[1], -vector[0]


def vertices_to_edges(vertices):
    """Find the edges of the vertices.

    Args:
        vertices (list): A list of the vertices.

    Returns:
        list: A list of the edges of the vertices as vectors.
    """
    return [edge_direction(vertices[i], vertices[(i + 1) % len(vertices)])
            for i in range(2)]


def project(vertices, axis):
    """Calculate the projection of the vertices along the axis.

    Args:
        vertices (list): A list of the vertices.
        axis (list): The axis along which to project the vertices.

    Returns:
        list: A vector showing how much of the vertices lies along the axis.
    """
    dots = [dot(vertex, axis) for vertex in vertices]
    return [min(dots), max(dots)]


def overlap(projection1, projection2):
    """Determine if two projections overlap.

    Args:
        projection1 (list): The first projection.
        projection2 (list): The second projection.

    Returns:
        bool: A boolean indicating if the two projections overlap.
    """
    return min(projection1) <= max(projection2) and \
           min(projection2) <= max(projection1)


def separating_axis_theorem(vertices_a, vertices_b):
    """Determine if two polygons intersect using the Separating Axis Theorem.

    Args:
        vertices_a (list): The vertices of the first polygon.
        vertices_b (list): The vertices of the second polygon.

    Returns:
        bool: A boolean indicating if the two polygons intersect.
    """
    edges = vertices_to_edges(vertices_a) + vertices_to_edges(vertices_b)
    axes = [normalize(orthogonal(edge)) for edge in edges]

    for axis in axes:
        projection_a = project(vertices_a, axis)
        projection_b = project(vertices_b, axis)

        overlapping = overlap(projection_a, projection_b)

        if not overlapping:
            return False

    return True


def main():
    """A simple test of the Separating Axis Theorem.
    """    
    a_vertices = [(0, 0), (70, 0), (0, 70)]
    b_vertices = [(70, 70), (150, 70), (70, 150)]
    c_vertices = [(30, 30), (150, 70), (70, 150)]

    print(separating_axis_theorem(a_vertices, b_vertices))
    print(separating_axis_theorem(a_vertices, c_vertices))
    print(separating_axis_theorem(b_vertices, c_vertices))


if __name__ == "__main__":
    main()