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
    _wavespeed = 0.05
    _viewport = [0.0, 0.0, 0.0, 0.0]
    _update_viewport = False
    _fullscreen = False    # current display
    _toggle_fullscreen = None     # callback for onclick function

    def __init__(self, get_data, get_2d_layout, toggle_fullscreen, settings):
        super().__init__()

        self._get_data = get_data
        self._toggle_fullscreen = toggle_fullscreen

        # initialise data structures
        # electrodes and their states
        layout = get_2d_layout()
        self._x_positions = np.zeros(len(layout), dtype=np.float32)
        self._y_positions = np.zeros(len(layout), dtype=np.float32)
        self._red_values = np.zeros(len(layout), dtype=np.float32)
        self._green_values = np.zeros(len(layout), dtype=np.float32)
        self._blue_values = np.zeros(len(layout), dtype=np.float32)
        for n in range(0, len(layout)):
            self._x_positions[n] = 0.8 * layout[n][0]
            self._y_positions[n] = 0.8 * layout[n][1]
            self._red_values[n] = layout[n][2]
            self._green_values[n] = layout[n][3]
            self._blue_values[n] = layout[n][4]

        self._speeds = 0.1 * np.random.rand(len(self._x_positions)) * np.random.rand(len(self._x_positions))     # this is the actual data
        self._counters = np.random.rand(len(self._x_positions))     # counters for starting new circles

        # the queue with circles for the screen
        self._x_numbers = np.zeros(100, dtype=np.float32)     # x position on screen
        self._y_numbers = np.zeros(100, dtype=np.float32)     # y position on screen
        self._radii     = np.zeros(100, dtype=np.float32)     # radius on screen
        self._red       = np.zeros(100, dtype=np.float32)      # red of rgb on screen
        self._green     = np.zeros(100, dtype=np.float32)      # green of rgb on screen
        self._blue      = np.zeros(100, dtype=np.float32)      # blue of rgb on screen

        self.set_parameters(settings)

    def set_parameters(self, settings):

        if self._timer:
            self._timer.stop()

        self._wavespeed = 0.001 * settings['wavespeed']
        self._wavefrequency = 0.01 * settings['wavefrequency']

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
        shader_code = (" #version 400 core\n"
                       " layout (location = 0) in vec2 xyCoords; "
                       " layout (location = 1) in float radius; "
                       " layout (location = 2) in vec3 vxColour; "
                       " out vec4 colourSizeV; "
                       " out vec2 pointCenterV; "
                       " void main() { "
                       "     gl_Position = vec4(xyCoords, 0.0, 1.0); "
                       "     colourSizeV = vec4(vxColour, radius); "
                       "     pointCenterV = xyCoords; "
                       " } ")
        gl.glShaderSource(vertex_shader_id, shader_code)
        gl.glCompileShader(vertex_shader_id)
        if gl.glGetShaderiv(vertex_shader_id, gl.GL_COMPILE_STATUS) == gl.GL_FALSE:
            print(f"Error creating radiant ripples vertex shader: {gl.glGetShaderInfoLog(vertex_shader_id)}.")

        # the geometry shader
        geometry_shader_id = gl.glCreateShader(gl.GL_GEOMETRY_SHADER)
        shader_code = (" #version 400 core\n"
                       " layout(points) in; "
                       " layout(triangle_strip, max_vertices=6) out; "
                       " in vec4 colourSizeV [ ]; "
                       " in vec2 pointCenterV [ ]; "
                       " out vec4 colourSize; "
                       " out vec2 pointCenter; "
                       " void main() { "
                       "     colourSize = colourSizeV[0]; "
                       "     pointCenter = pointCenterV[0]; "
                       "     gl_Position = gl_in[0].gl_Position + vec4(-0.5, -0.5, 0.0, 0.0); EmitVertex(); "
                       "     gl_Position = gl_in[0].gl_Position + vec4(-0.5, 0.5, 0.0, 0.0); EmitVertex(); "
                       "     gl_Position = gl_in[0].gl_Position + vec4(0.5, 0.5, 0.0, 0.0); EmitVertex(); "
                       "     gl_Position = gl_in[0].gl_Position + vec4(-0.5, -0.5, 0.0, 0.0); EmitVertex(); "
                       "     gl_Position = gl_in[0].gl_Position + vec4(0.5, -0.5, 0.0, 0.0); EmitVertex(); "
                       "     gl_Position = gl_in[0].gl_Position + vec4(0.5, 0.5, 0.0, 0.0); EmitVertex(); "
                       "     EndPrimitive(); "
                       " } ")
        gl.glShaderSource(geometry_shader_id, shader_code)
        gl.glCompileShader(geometry_shader_id)
        if gl.glGetShaderiv(geometry_shader_id, gl.GL_COMPILE_STATUS) == gl.GL_FALSE:
            print(f"Error creating radiant ripples geometry shader: {gl.glGetShaderInfoLog(geometry_shader_id)}.")

        # the fragment shader
        fragment_shader_id = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        shader_code = (" #version 400 core\n"
                       " in vec4 colourSize; \n"
                       " in vec2 pointCenter; \n"
                       " uniform float iResolution; \n"
                       " uniform vec2 iWindowCorner; "
                       " out vec4 fragColour; \n"
                       " void main() { \n"
                       "     vec2 uv = 2.0*(gl_FragCoord.xy-iWindowCorner)/iResolution - 1.0 - pointCenter.xy; \n"
                       "     float s = colourSize[3]; \n"
                       "     float w = 100.0 / pow(s,2.0); \n"
                       "     float r = 2.5*length(uv); \n"
                       "     float k = pow((s-r), 2.0); \n"
                       "     float c = pow(2.0, -k*w); \n"
                       "     fragColour = vec4(c * colourSize[0], \n"
                       "                       c * colourSize[1], \n"
                       "                       c * colourSize[2], c/pow(2.0, 5.0*s)); \n"
                       " } ")
        gl.glShaderSource(fragment_shader_id, shader_code)
        gl.glCompileShader(fragment_shader_id)
        if gl.glGetShaderiv(fragment_shader_id, gl.GL_COMPILE_STATUS) == gl.GL_FALSE:
            print(f"Error creating radiant ripples fragment shader: {gl.glGetShaderInfoLog(fragment_shader_id)}.")

        # the shader program, linking both shaders
        self._shader_program_id = gl.glCreateProgram()
        gl.glAttachShader(self._shader_program_id, vertex_shader_id)
        gl.glAttachShader(self._shader_program_id, geometry_shader_id)
        gl.glAttachShader(self._shader_program_id, fragment_shader_id)
        gl.glLinkProgram(self._shader_program_id)
        if gl.glGetProgramiv(self._shader_program_id, gl.GL_LINK_STATUS) == gl.GL_FALSE:
            print(f"Error linking radiant ripples shaders: {gl.glGetProgramInfoLog(self._shader_program_id)}.")
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
        self._radii += self._wavespeed

        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self._vertices.nbytes, self._vertices)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        if self._update_viewport:
            gl.glViewport(0, 0, self._viewport[2], self._viewport[3])

        # actual drawing
        loc = gl.glGetUniformLocation(self._shader_program_id, "iResolution")
        gl.glUniform1f(loc, min(self._viewport[2], self._viewport[3]))
        loc = gl.glGetUniformLocation(self._shader_program_id, "iWindowCorner")
        gl.glUniform2f(loc, self._viewport[0], self._viewport[1])
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(self._x_numbers))
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"glDrawArrays error: {error}")

        # next step
        data = self._get_data()[:,-1]
        data -= data.min()
        data /= data.max() or 1.0
        self._counters -= self._wavefrequency * data
        for n in range(0, len(self._counters)):
            if self._counters[n] < 0.0:
                self._x_numbers = np.roll(self._x_numbers, 1)
                self._y_numbers = np.roll(self._y_numbers, 1)
                self._red = np.roll(self._red, 1)
                self._green = np.roll(self._green, 1)
                self._blue = np.roll(self._blue, 1)
                self._radii = np.roll(self._radii, 1)
                self._x_numbers[0] = self._x_positions[n]
                self._y_numbers[0] = self._y_positions[n]
                self._radii[0] = 0.0
                self._red[0] = self._red_values[n]
                self._green[0] = self._green_values[n]
                self._blue[0] = self._blue_values[n]
                self._counters[n] = 1.0

    def resizeGL(self, width, height):
        size = min(width, height)
        x = (width - size) // 2
        y = (height - size) // 2
        gl.glUseProgram(self._shader_program_id)
        gl.glViewport(0, 0, width, height)    # ignore x and y in the actual viewport call
        self._viewport = [x, y, width, height]      # but use it in calculations
        self._update_viewport = True
        print(f"glViewport set to x={0}, y={0}, width={width}, height={height}")

    def start(self):
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.start(30)

    def mouseReleaseEvent(self, dummy):
        self._fullscreen = not self._fullscreen
        self._toggle_fullscreen(self._fullscreen)
