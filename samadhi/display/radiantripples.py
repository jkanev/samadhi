# -*- coding:utf-8 -*-
#!/usr/bin/python3
import ctypes

from OpenGL import GL as gl
from PyQt6 import QtCore, QtOpenGLWidgets, QtOpenGL
from PyQt6.QtGui import QSurfaceFormat
import numpy as np

class OpenGLRadiantRipples(QtOpenGLWidgets.QOpenGLWidget):

    _get_data = False
    _x_numbers = False
    _y_numbers = False
    _vertices = False
    _red = False
    _green = False
    _blue = False
    _timer = False
    _shader_program_id = 0
    _vertex_buffer = None  # VBO
    _vertex_array = None  # VAO
    _counter = 0.0
    _viewport = [0.0, 0.0, 0.0, 0.0]
    _update_viewport = False
    _fullscreen = False    # current display
    _toggle_fullscreen = None     # callback for onclick function

    # inside:  red / yellow / blue / orange / green
    # outside: orange / green / red / yellow / blue
    red = np.array([0.8, 0.0, 0.0])
    orange = np.array([1.0, 0.5, 0.0])
    yellow = np.array([0.6, 0.6, 0.0])
    green = np.array([0.0, 0.6, 0.0])
    turquoise = np.array([0.0, 0.6, 0.6])
    blue = np.array([0.0, 0.0, 1.0])
    purple = np.array([0.8, 0.0, 0.4])
    black = np.array([0.0, 0.0, 0.0])
    dark = 0.3
    _data_colours = [[red, dark*green],
                     [yellow, dark*blue],
                     [orange, dark*turquoise],
                     [green, dark*red],
                     [blue, dark*yellow]]
    _rotations = [0, 0, 0, 0, 0]

    def __init__(self, get_data, toggle_fullscreen, settings):
        super().__init__()

        self._get_data = get_data
        self._toggle_fullscreen = toggle_fullscreen

        # initialise data structures
        self._x_numbers = np.array([-0.1, -0.1, -0.1, 0.0, 0.0,  0.0, 0.1, 0.1,  0.1, -0.1, -0.1, -0.1, 0.0, 0.0,  0.0, 0.1, 0.1,  0.1, ], dtype=np.float32)
        self._y_numbers = np.array([ 0.1,  0.0, -0.1, 0.1, 0.0, -0.1, 0.1, 0.0, -0.1,  0.1,  0.0, -0.1, 0.1, 0.0, -0.1, 0.1, 0.0, -0.1, ], dtype=np.float32)
        self._radii     = np.array([ 0.0,  0.1,  0.2, 0.3,  0.2,  0.7, 0.6, 0.4,  0.1, 0.8, 0.5,  0.3, 0.7, 0.2,  0.0, 0.9, 0.5,  0.8, ], dtype=np.float32)
        self._red       = np.array([ 0.2,  0.3,  0.4, 0.2,  0.3,  0.4, 0.5, 0.6,  0.7, 0.5, 0.6,  0.7, 0.8, 0.9,  1.0, 0.8, 0.9,  1.0, ], dtype=np.float32)
        self._green     = np.array([ 0.0,  0.2,  0.4, 0.8, 1.0,  0.8, 0.6, 0.4,  0.2, 0.0,  0.2,  0.4, 0.8, 1.0,  0.8, 0.6, 0.4,  0.2, ], dtype=np.float32)
        self._blue      = np.array([ 1.0,  0.9,  1.0,  0.9,  0.8, 0.7, 0.8, 0.7, 0.6,  0.5, 0.6,  0.5, 0.4, 0.3,  0.4, 0.3,  0.2, 0.2, ], dtype=np.float32)

        self.set_parameters(settings)


    def set_parameters(self, settings):

        if self._timer:
            self._timer.stop()

        # restart timer again
        if self._timer:
            self._timer.start(30)

    def initializeGL(self):

        print(f"GL_VENDOR: {gl.glGetString(gl.GL_VENDOR)}")
        print(f"GL_RENDERER: {gl.glGetString(gl.GL_RENDERER)}")
        print(f"GL_VERSION: {gl.glGetString(gl.GL_VERSION)}")
        print(f"GL_SHADING_LANGUAGE_VERSION: {gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)}")

        # the vertex shader
        vertex_shader_id = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        shader_code = (" #version 330 core\n"
                       " layout (location = 0) in vec2 xyCoords; "
                       " layout (location = 1) in float radius; "
                       " layout (location = 2) in vec3 vxColour; "
                       " out vec4 colourSize; "
                       " void main() { "
                       "     gl_Position = vec4(xyCoords, 0.0, 1.0); "
                       "     colourSize = vec4(vxColour, radius); "
                       " } ")
        gl.glShaderSource(vertex_shader_id, shader_code)
        gl.glCompileShader(vertex_shader_id)
        if gl.glGetShaderiv(vertex_shader_id, gl.GL_COMPILE_STATUS) == gl.GL_FALSE:
            print(f"Error creating radiant ripples vertex shader: {gl.glGetShaderInfoLog(vertex_shader_id)}.")

        # the fragment shader
        fragment_shader_id = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        shader_code = (" #version 330 core\n"
                       " in vec4 colourSize; "
                       " out vec4 fragColour; "
                       " void main() { "
                       "     vec2 uv = gl_PointCoord * 2.0 - 1.0;"
                       "     float s = colourSize[3]; "
                       "     float w = 100.0 / pow(s,2.0); "
                       "     float r = length(uv); "
                       "     float k = pow((s-r), 2.0); "
                       "     float c = pow(2.0, -k*w); "
                       "     fragColour = vec4(c*colourSize[0], c*colourSize[1], c*colourSize[2], c/pow(2.0, 5.0*s)); "
                       " } ")
        gl.glShaderSource(fragment_shader_id, shader_code)
        gl.glCompileShader(fragment_shader_id)
        if gl.glGetShaderiv(fragment_shader_id, gl.GL_COMPILE_STATUS) == gl.GL_FALSE:
            print(f"Error creating radiant ripples fragment shader: {gl.glGetShaderInfoLog(fragment_shader_id)}.")

        # the shader program, linking both shaders
        self._shader_program_id = gl.glCreateProgram()
        gl.glAttachShader(self._shader_program_id, vertex_shader_id)
        gl.glAttachShader(self._shader_program_id, fragment_shader_id)
        gl.glLinkProgram(self._shader_program_id)
        if gl.glGetProgramiv(self._shader_program_id, gl.GL_LINK_STATUS) == gl.GL_FALSE:
            print("Error linking radiant ripples shaders.")
        if not gl.glIsProgram(self._shader_program_id):
            print(f"Error: Shader program {self._shader_program_id} is not valid!")

        # declare the buffer to be a vertex array
        self._vertices = np.column_stack((self._x_numbers, self._y_numbers, self._radii, self._red, self._green, self._blue)).ravel()

        # create the VAO
        self._vertex_array = QtOpenGL.QOpenGLVertexArrayObject()
        self._vertex_array.create()
        self._vertex_array.bind()

        self._vertex_buffer = QtOpenGL.QOpenGLBuffer()
        self._vertex_buffer.create()
        self._vertex_buffer.bind()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self._vertices.nbytes, self._vertices, gl.GL_DYNAMIC_DRAW)
        size = self._vertices.itemsize
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 6*size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 1, gl.GL_FLOAT, gl.GL_FALSE, 6*size, ctypes.c_void_p(2 * size))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 6*size, ctypes.c_void_p(3 * size))
        gl.glEnableVertexAttribArray(2)
        gl.glUseProgram(self._shader_program_id)
        gl.glPointSize(2000)

    def paintGL(self):

        # convert to x/y/colour data and push to graphic card
        self._vertices = np.column_stack((self._x_numbers, self._y_numbers, self._radii, self._red, self._green, self._blue)).ravel()

        gl.glUseProgram(self._shader_program_id)
        self._vertex_array.bind()
        self._vertex_buffer.bind()
        self._radii += 0.01
        for n in range(0, len(self._radii)):
            self._radii[n] = self._radii[n] < 0.9 and self._radii[n] or 0.0
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self._vertices.nbytes, self._vertices)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        if self._update_viewport:
            gl.glViewport(*self._viewport)

        # actual drawing
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA);
        gl.glDrawArrays(gl.GL_POINTS, 0, len(self._x_numbers))
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"glDrawArrays error: {error}")

    def resizeGL(self, width, height):
        size = min(width, height)
        x = (width - size) // 2
        y = (height - size) // 2
        gl.glUseProgram(self._shader_program_id)
        gl.glViewport(x, y, size, size)
        self._viewport = [x, y, size, size]
        self._update_viewport = True
        print(f"glViewport set to x={x}, y={y}, width={size}, height={size}")

    def start(self):
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.start(30)

    def mouseReleaseEvent(self, dummy):
        self._fullscreen = not self._fullscreen
        self._toggle_fullscreen(self._fullscreen)
