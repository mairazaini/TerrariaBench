#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Terminal resolution in characters.
// 120x40 fits standard terminal windows with proper character aspect ratio.
#define X_PIXELS 120
#define Y_PIXELS 40

// View angles in radians:
#define VIEW_WIDTH 1.0f
#define VIEW_HEIGHT 0.7f

#define BLOCKS_X 20
#define BLOCKS_Y 20
#define BLOCKS_Z 10

#define EYE_HEIGHT 1.5f

#define START_X 5.0f
#define START_Y 5.0f
#define START_Z (4.0f + EYE_HEIGHT)

#define BLOCK_BORDER_SIZE 0.03f
#define BLOCK_INFILL_CHAR '@'
#define BLOCK_BORDER_CHAR '.'

typedef struct Vector {
    float x;
    float y;
    float z;
} vect;

typedef struct Vector2 {
    float psi; // pitch (tilt up / down)
    float phi; // yaw (turn left / right)
} vect2;

typedef struct Vector_Vector2 {
    vect pos;
    vect2 view;
} player_pos_view;

// Windows asynchronous key check
int key_is_pressed(int vKey) {
    return (GetAsyncKeyState(vKey) & 0x8000) != 0;
}

vect vect_add(vect v1, vect v2) {
    vect res = { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
    return res;
}

vect vect_scale(float s, vect v) {
    vect res = { s * v.x, s * v.y, s * v.z };
    return res;
}

void vect_normalize(vect *v) {
    float len = sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);
    if (len > 0.00001f) {
        v->x /= len;
        v->y /= len;
        v->z /= len;
    }
}

vect vect_sub(vect v1, vect v2) {
    vect v3 = vect_scale(-1.0f, v2);
    return vect_add(v1, v3);
}

player_pos_view init_posview(void) {
    player_pos_view posview;
    posview.pos.x = START_X;
    posview.pos.y = START_Y;
    posview.pos.z = START_Z;
    posview.view.psi = 0.0f;
    posview.view.phi = 0.0f;
    return posview;
}

char **init_picture(void) {
    char **picture = malloc(sizeof(char *) * Y_PIXELS);
    for (int i = 0; i < Y_PIXELS; ++i) {
        picture[i] = malloc(sizeof(char) * X_PIXELS);
    }
    return picture;
}

char ***init_blocks(void) {
    char ***blocks = malloc(sizeof(char **) * BLOCKS_Z);
    for (int i = 0; i < BLOCKS_Z; ++i) {
        blocks[i] = malloc(sizeof(char *) * BLOCKS_Y);
        for (int j = 0; j < BLOCKS_Y; ++j) {
            blocks[i][j] = malloc(sizeof(char) * BLOCKS_X);
            for (int k = 0; k < BLOCKS_X; ++k) {
                blocks[i][j][k] = ' ';
            }
        }
    }
    return blocks;
}

vect angles_to_vect(vect2 angles) {
    vect res;
    res.x = cosf(angles.psi) * cosf(angles.phi);
    res.y = cosf(angles.psi) * sinf(angles.phi);
    res.z = sinf(angles.psi);
    return res;
}

vect **init_directions(void) {
    vect **dir = malloc(sizeof(vect *) * Y_PIXELS);
    for (int i = 0; i < Y_PIXELS; ++i) {
        dir[i] = malloc(sizeof(vect) * X_PIXELS);
    }
    return dir;
}

void update_directions(vect **dir, vect2 view) {
    view.psi -= VIEW_HEIGHT / 2.0f;
    vect screen_down = angles_to_vect(view);
    view.psi += VIEW_HEIGHT;
    vect screen_up = angles_to_vect(view);
    view.psi -= VIEW_HEIGHT / 2.0f;
    view.phi -= VIEW_WIDTH / 2.0f;
    vect screen_left = angles_to_vect(view);
    view.phi += VIEW_WIDTH;
    vect screen_right = angles_to_vect(view);
    view.phi -= VIEW_WIDTH / 2.0f;

    vect screen_mid_vert = vect_scale(0.5f, vect_add(screen_up, screen_down));
    vect screen_mid_hor = vect_scale(0.5f, vect_add(screen_left, screen_right));

    vect mid_to_left = vect_sub(screen_left, screen_mid_hor);
    vect mid_to_up = vect_sub(screen_up, screen_mid_vert);

    for (int y_pix = 0; y_pix < Y_PIXELS; ++y_pix) {
        for (int x_pix = 0; x_pix < X_PIXELS; ++x_pix) {
            vect tmp = vect_add(vect_add(screen_mid_hor, mid_to_left), mid_to_up);
            tmp = vect_sub(tmp, vect_scale(((float)x_pix / (X_PIXELS - 1)) * 2.0f, mid_to_left));
            tmp = vect_sub(tmp, vect_scale(((float)y_pix / (Y_PIXELS - 1)) * 2.0f, mid_to_up));
            vect_normalize(&tmp);
            dir[y_pix][x_pix] = tmp;
        }
    }
}

int ray_outside(vect pos) {
    if (pos.x >= BLOCKS_X || pos.y >= BLOCKS_Y || pos.z >= BLOCKS_Z) return 1;
    if (pos.x < 0 || pos.y < 0 || pos.z < 0) return 1;
    return 0;
}

static inline float min_val(float a, float b) {
    return (a < b) ? a : b;
}

int on_block_border(vect pos) {
    int cnt = 0;
    if (fabsf(pos.x - roundf(pos.x)) < BLOCK_BORDER_SIZE) cnt++;
    if (fabsf(pos.z - roundf(pos.z)) < BLOCK_BORDER_SIZE) cnt++;
    if (fabsf(pos.y - roundf(pos.y)) < BLOCK_BORDER_SIZE) cnt++;
    return cnt >= 2;
}

char raytrace(vect pos, vect dir, char ***blocks) {
    float eps = 0.01f;
    while (!ray_outside(pos)) {
        char c = blocks[(int)pos.z][(int)pos.y][(int)pos.x];
        if (c != ' ') {
            return on_block_border(pos) ? BLOCK_BORDER_CHAR : c;
        }
        float dist = 2.0f;
        if (dir.x > eps)       dist = min_val(dist, ((int)(pos.x + 1) - pos.x) / dir.x);
        else if (dir.x < -eps) dist = min_val(dist, ((int)pos.x - pos.x) / dir.x);

        if (dir.y > eps)       dist = min_val(dist, ((int)(pos.y + 1) - pos.y) / dir.y);
        else if (dir.y < -eps) dist = min_val(dist, ((int)pos.y - pos.y) / dir.y);

        if (dir.z > eps)       dist = min_val(dist, ((int)(pos.z + 1) - pos.z) / dir.z);
        else if (dir.z < -eps) dist = min_val(dist, ((int)pos.z - pos.z) / dir.z);

        pos = vect_add(pos, vect_scale(dist + eps, dir));
    }
    return ' ';
}

void get_picture(char **picture, vect **directions, player_pos_view posview, char ***blocks) {
    update_directions(directions, posview.view);
    for (int y = 0; y < Y_PIXELS; ++y) {
        for (int x = 0; x < X_PIXELS; ++x) {
            picture[y][x] = raytrace(posview.pos, directions[y][x], blocks);
        }
    }
}

// Single-buffer rendering on Windows
static HANDLE hConsoleOut;
static char frame_buffer[(X_PIXELS + 1) * Y_PIXELS + 1];

void init_console(void) {
    hConsoleOut = GetStdHandle(STD_OUTPUT_HANDLE);

    // Hide console cursor to prevent blinking artifacts
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(hConsoleOut, &cursorInfo);
    cursorInfo.bVisible = FALSE;
    SetConsoleCursorInfo(hConsoleOut, &cursorInfo);

    // Enable Virtual Terminal Processing if supported
    DWORD mode = 0;
    if (GetConsoleMode(hConsoleOut, &mode)) {
        SetConsoleMode(hConsoleOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
}

void draw_ascii(char **picture) {
    COORD origin = { 0, 0 };
    SetConsoleCursorPosition(hConsoleOut, origin);

    int idx = 0;
    for (int i = 0; i < Y_PIXELS; ++i) {
        for (int j = 0; j < X_PIXELS; ++j) {
            frame_buffer[idx++] = picture[i][j];
        }
        frame_buffer[idx++] = '\n';
    }
    frame_buffer[idx] = '\0';

    fwrite(frame_buffer, 1, idx, stdout);
    fflush(stdout);
}

void update_posview(player_pos_view *posview, char*** blocks) {
    float move_eps = 0.15f;
    float tilt_eps = 0.03f;
    int x = (int)posview->pos.x;
    int y = (int)posview->pos.y;

    // Check bounds before checking collisions
    if (x >= 0 && x < BLOCKS_X && y >= 0 && y < BLOCKS_Y) {
        int z_in = (int)(posview->pos.z - EYE_HEIGHT + 0.01f);
        if (z_in >= 0 && z_in < BLOCKS_Z && blocks[z_in][y][x] != ' ') {
            posview->pos.z += 1.0f;
        }
        int z_below = (int)(posview->pos.z - EYE_HEIGHT - 0.01f);
        if (z_below >= 0 && z_below < BLOCKS_Z && blocks[z_below][y][x] == ' ') {
            posview->pos.z -= 1.0f;
        }
    }

    // WASD: look around
    if (key_is_pressed('W')) posview->view.psi += tilt_eps;
    if (key_is_pressed('S')) posview->view.psi -= tilt_eps;
    if (key_is_pressed('A')) posview->view.phi -= tilt_eps;
    if (key_is_pressed('D')) posview->view.phi += tilt_eps;

    vect direction = angles_to_vect(posview->view);

    // Arrow keys: move and strafe
    if (key_is_pressed(VK_UP)) {
        posview->pos.x += move_eps * direction.x;
        posview->pos.y += move_eps * direction.y;
    }
    if (key_is_pressed(VK_DOWN)) {
        posview->pos.x -= move_eps * direction.x;
        posview->pos.y -= move_eps * direction.y;
    }
    if (key_is_pressed(VK_LEFT)) {
        posview->pos.x += move_eps * direction.y;
        posview->pos.y -= move_eps * direction.x;
    }
    if (key_is_pressed(VK_RIGHT)) {
        posview->pos.x -= move_eps * direction.y;
        posview->pos.y += move_eps * direction.x;
    }
}

vect get_current_block(player_pos_view posview, char ***blocks) {
    float eps = 0.01f;
    vect pos = posview.pos;
    vect dir = angles_to_vect(posview.view);
    while (!ray_outside(pos)) {
        char c = blocks[(int)pos.z][(int)pos.y][(int)pos.x];
        if (c != ' ') {
            return pos;
        }
        float dist = 2.0f;
        if (dir.x > eps)       dist = min_val(dist, ((int)(pos.x + 1) - pos.x) / dir.x);
        else if (dir.x < -eps) dist = min_val(dist, ((int)pos.x - pos.x) / dir.x);

        if (dir.y > eps)       dist = min_val(dist, ((int)(pos.y + 1) - pos.y) / dir.y);
        else if (dir.y < -eps) dist = min_val(dist, ((int)pos.y - pos.y) / dir.y);

        if (dir.z > eps)       dist = min_val(dist, ((int)(pos.z + 1) - pos.z) / dir.z);
        else if (dir.z < -eps) dist = min_val(dist, ((int)pos.z - pos.z) / dir.z);

        pos = vect_add(pos, vect_scale(dist + eps, dir));
    }
    return pos;
}

void place_block(vect pos, char ***blocks, char block) {
    int x = (int)pos.x, y = (int)pos.y, z = (int)pos.z;
    float dists[6] = {
        fabsf(x + 1 - pos.x),
        fabsf(pos.x - x),
        fabsf(y + 1 - pos.y),
        fabsf(pos.y - y),
        fabsf(z + 1 - pos.z),
        fabsf(pos.z - z)
    };
    int min_idx = 0;
    float mindist = dists[0];
    for (int i = 1; i < 6; ++i) {
        if (dists[i] < mindist) {
            mindist = dists[i];
            min_idx = i;
        }
    }
    int nx = x, ny = y, nz = z;
    switch (min_idx) {
        case 0: nx++; break;
        case 1: nx--; break;
        case 2: ny++; break;
        case 3: ny--; break;
        case 4: nz++; break;
        case 5: nz--; break;
    }
    if (nx >= 0 && nx < BLOCKS_X && ny >= 0 && ny < BLOCKS_Y && nz >= 0 && nz < BLOCKS_Z) {
        blocks[nz][ny][nx] = block;
    }
}

int main(void) {
    init_console();

    char **picture = init_picture();
    char ***blocks = init_blocks();
    vect **directions = init_directions();
    player_pos_view posview = init_posview();

    for (int x = 0; x < BLOCKS_X; ++x) {
        for (int y = 0; y < BLOCKS_Y; ++y) {
            for (int z = 0; z < 4; ++z) {
                blocks[z][y][x] = BLOCK_INFILL_CHAR;
            }
        }
    }

    system("cls");

    int place_block_allowed = 1;
    while (1) {
        // Press ESC or Q to quit
        if (key_is_pressed(VK_ESCAPE) || key_is_pressed('Q')) {
            break;
        }

        update_posview(&posview, blocks);

        vect current_block = get_current_block(posview, blocks);
        int have_current_block = !ray_outside(current_block);
        int current_block_x = (int)current_block.x;
        int current_block_y = (int)current_block.y;
        int current_block_z = (int)current_block.z;
        char current_block_c = ' ';
        int removed_block = 0;

        if (have_current_block) {
            current_block_c = blocks[current_block_z][current_block_y][current_block_x];
            blocks[current_block_z][current_block_y][current_block_x] = 'o';

            if (place_block_allowed && key_is_pressed(VK_SPACE)) {
                if (key_is_pressed(VK_SHIFT)) {
                    blocks[current_block_z][current_block_y][current_block_x] = ' ';
                    removed_block = 1;
                } else {
                    place_block(current_block, blocks, BLOCK_INFILL_CHAR);
                }
                place_block_allowed = 0;
            } else if (!key_is_pressed(VK_SPACE)) {
                place_block_allowed = 1;
            }
        } else if (!key_is_pressed(VK_SPACE)) {
            place_block_allowed = 1;
        }

        get_picture(picture, directions, posview, blocks);

        if (have_current_block && !removed_block) {
            blocks[current_block_z][current_block_y][current_block_x] = current_block_c;
        }

        draw_ascii(picture);

        // Cap to ~60 FPS
        Sleep(16);
    }

    // Restore cursor before exit
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(hConsoleOut, &cursorInfo);
    cursorInfo.bVisible = TRUE;
    SetConsoleCursorInfo(hConsoleOut, &cursorInfo);

    return 0;
}