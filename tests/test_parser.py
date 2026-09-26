import pytest
from src.glucotwin.data.parser import DataParser
import xml.etree.ElementTree as ET

def test_parse_synthetic_xml():
    xml = '''<?xml version="1.0" ?>
    <patient id="559">
        <glucose_level>
            <event ts="01-01-2026 08:00:00" value="120" />
        </glucose_level>
        <basal>
            <event ts="01-01-2026 08:00:00" value="0.5" />
        </basal>
    </patient>
    '''
    records = DataParser.parse_synthetic_xml(xml, "559")
    assert len(records["cgm"]) == 1
    assert records["cgm"][0].glucose == 120.0
    assert len(records["insulin"]) == 1
    assert records["insulin"][0].basal_insulin == 0.5
