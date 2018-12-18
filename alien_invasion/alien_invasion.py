import sys
import pygame

from pygame.sprite import Group

from settings import Settings
from ship import Ship
from ycy import Ycy
from bullet import Bullet
import game_functions as gf

def run_game():
	pygame.init()
	ai_settings = Settings()
	screen = pygame.display.set_mode((ai_settings.screen_width,ai_settings.screen_height))
	pygame.display.set_caption("Alien Invasion")
	ai_ship = Ship(ai_settings, screen)
	ai_ycy = Ycy(screen)

	bullets = Group()

	while True:
		gf.check_events(ai_settings, screen, ai_ship, bullets)
		ai_ship.update()
		gf.update_bullets(bullets)
		gf.update_screen(ai_settings,screen,ai_ship,ai_ycy,bullets)

run_game()