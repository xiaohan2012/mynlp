import os
from mynlp.kb.freebase import FreebaseAnnotator
from nose.tools import (assert_equal, with_setup)
from pathlib import Path

cur_dir = str(Path(__file__).resolve().parent)

VIACOM = {"/common/topic/description": [], "name": "Viacom", "industry": ["Media", "Mass media"], "/common/topic/official_website": ["http://www.viacom.com"], "/common/topic/alias": ["Viacom Inc.", "NEW VIACOM CORP"], "type": "/business/business_operation"}
VIACOM2 = {"/common/topic/description": [], "name": "Viacom2", "industry": ["Halo", "Hola"], "/common/topic/official_website": ["http://www.viacom2.com"], "/common/topic/alias": ["Viacom Inc."], "type": "/business/business_operation"}
APPLE = {"/common/topic/description": ["Apple Inc., (NASDAQ: AAPL) formerly Apple Computer Inc., is an American multinational corporation which designs and manufactures consumer electronics and software products. The company's best-known hardware products include Macintosh computers, the iPod and the iPhone. Apple software includes the Mac OS X operating system, the iTunes media browser, the iLife suite of multimedia and creativity software, the iWork suite of productivity software, and Final Cut Studio, a suite of professional audio and film-industry software products. The company operates more than 250 retail stores in nine countries and an online store where hardware and software products are sold."], "name": "Apple Inc.", "industry": ["Computer hardware", "Software", "Consumer electronics", "Technology", "Electronic Computer Manufacturing", "Digital distribution"], "/common/topic/official_website": ["http://www.apple.com/"], "/common/topic/alias": ["Apple Computer Company", "Apple Computers", "Apple Computer, Inc."], "type": "/business/business_operation"}

FREEBASE_DUMP_PATH = cur_dir + '/data/freebase.dump'
FREEBASE_RAW_PATH = cur_dir + '/data/freebase'

def setup():
    pass

def teardown():
    os.remove(FREEBASE_DUMP_PATH)

def test_normal():
    annotator = FreebaseAnnotator(FREEBASE_DUMP_PATH, FREEBASE_RAW_PATH)
    sent = 'Viacom Inc. and Apple Computers released a new phone .'.split()
    actual = annotator.annotate(sent)
    expected = [((0,1), VIACOM), ((0,1), VIACOM2), ((3,4), APPLE)]
    
    assert_equal(actual, expected)

def test_empty_output():
    annotator = FreebaseAnnotator(FREEBASE_DUMP_PATH, FREEBASE_RAW_PATH)
    sent = 'Halo Inc. released a new phone .'.split()

    actual = annotator.annotate(sent)
    expected = []
    
    assert_equal(actual, expected)
