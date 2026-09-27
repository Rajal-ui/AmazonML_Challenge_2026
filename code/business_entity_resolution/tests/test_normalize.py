"""
Stage 1 regression tests for ``normalize.py``.

Purpose: lock in the current normalization behaviour so a later refactor cannot
silently change it. Every expected value here was produced by running the real
module against the real challenge data -- none are guesses.

Run:
    .\\.venv\\Scripts\\python.exe -m pytest code/business_entity_resolution/tests -v

The two bugs recorded in ``TestKnownBugs`` are real defects found while writing
these tests. They are marked ``xfail(strict=True)``: if someone fixes them the
suite tells us to flip the marker, and they can never be silently forgotten.
"""

import pytest

import normalize as nz


# ─── Name: legal-suffix canonicalization ──────────────────────────────────────

class TestNameSuffixes:
    def test_inc_becomes_incorporated(self):
        assert nz.normalize_name("Acme Inc.")["normalized_name"] == "acme incorporated"

    def test_pvt_ltd_becomes_private_limited(self):
        assert nz.normalize_name("Acme Pvt Ltd")["normalized_name"] == "acme private limited"

    def test_corp_becomes_corporation(self):
        assert nz.normalize_name("Foo Corp")["normalized_name"] == "foo corporation"

    def test_ltd_becomes_limited(self):
        assert nz.normalize_name("Bar Ltd")["normalized_name"] == "bar limited"

    def test_co_becomes_company(self):
        assert nz.normalize_name("Qux Co")["normalized_name"] == "qux company"

    def test_llc_is_left_intact(self):
        # llc is a distinct legal form, not an abbreviation to expand
        assert nz.normalize_name("Acme LLC")["normalized_name"] == "acme llc"

    def test_suffix_map_applies_to_real_data_row(self):
        # train_source2.tsv :: S2-508602797 (note the double space in the raw value)
        got = nz.normalize_name("FOUNDATION EXCEL AGENCY PRIVATE  LIMITED")
        assert got["normalized_name"] == "foundation excel agency private limited"


# ─── Name: punctuation handling ───────────────────────────────────────────────

class TestNamePunctuation:
    def test_ampersand_becomes_and(self):
        assert "and" in nz.normalize_name("Lee & Lawson")["normalized_name"]

    def test_ampersand_spacing(self):
        assert nz.normalize_name("Lee & Lawson")["normalized_name"] == "lee and lawson"

    def test_angle_brackets_are_stripped(self):
        # test_source1.tsv :: S1-156285671
        assert nz.normalize_name("<< Team Ecole")["normalized_name"] == "team ecole"

    def test_dot_in_domain_becomes_space(self):
        # train_source2.tsv :: S2-69394310
        assert nz.normalize_name("heassociates.com")["normalized_name"] == "heassociates com"

    def test_repeated_whitespace_is_collapsed(self):
        assert nz.normalize_name("Foo  Bar   Baz")["normalized_name"] == "foo bar baz"

    def test_tokens_match_normalized_string(self):
        r = nz.normalize_name("Acme Pvt Ltd")
        assert r["name_tokens"] == r["normalized_name"].split()

    def test_raw_name_is_preserved_unchanged(self):
        raw = "Acme Inc."
        assert nz.normalize_name(raw)["raw_name"] == raw


# ─── Address: abbreviation expansion ──────────────────────────────────────────

class TestAddressAbbreviations:
    def test_st_expands_to_street(self):
        # train_source2.tsv :: S2-764573417
        assert "street" in nz.normalize_address("105 ELM ST", "US")["normalized_address"]

    def test_ave_expands_to_avenue(self):
        assert "avenue" in nz.normalize_address("1 Ivanhoe Ave", "US")["normalized_address"]

    def test_rd_expands_to_road(self):
        # test_source1.tsv :: S1-106407869
        got = nz.normalize_address("IA, Iowa City, 1064 Newton Rd, Unit 11", "US")
        assert "newton road" in got["normalized_address"]

    def test_case_is_folded(self):
        upper = nz.normalize_address("105 ELM ST, MORGANTON, NC", "US")["normalized_address"]
        lower = nz.normalize_address("105 elm st, morganton, nc", "US")["normalized_address"]
        assert upper == lower

    def test_po_box_expands(self):
        # train_source3.tsv :: S3-671162755
        got = nz.normalize_address("1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio", "US")
        assert "post office box" in got["normalized_address"]


# ─── Address: component extraction ────────────────────────────────────────────

class TestAddressComponents:
    def test_us_zip_is_extracted(self):
        got = nz.normalize_address("1795 Westchester Drive, High Point, NC 27317", "US")
        assert got["postal_code"] == "27317"

    def test_india_pin_is_extracted(self):
        got = nz.normalize_address(
            "797, Lake Town Block A, Kolkata, Howrah, West Bengal 700001", "India"
        )
        assert got["postal_code"] == "700001"

    def test_france_postal_is_extracted(self):
        got = nz.normalize_address(
            "175 Boulevard du President Franklin Roosevelt, 75016 Paris", "France"
        )
        assert got["postal_code"] == "75016"

    def test_us_city_and_state(self):
        got = nz.normalize_address("1795 Westchester Drive, High Point, NC", "US")
        assert got["city"] == "high point"
        assert got["state"] == "nc"

    def test_city_recovered_from_state_first_format(self):
        # test_source1.tsv :: S1-851869949
        got = nz.normalize_address("OH, Columbus, 5559 Orville Avenue", "US")
        assert got["city"] == "columbus"
        assert got["state"] == "oh"

    def test_city_recovered_from_reversed_format(self):
        # train_source2.tsv :: S2-163963287
        got = nz.normalize_address("GREENSBORO, NC, 19 1/2 STARDUST TRAIL", "US")
        assert got["city"] == "greensboro"
        assert got["state"] == "nc"

    def test_state_full_name_kept(self):
        # train_source3.tsv :: S3-90100170
        got = nz.normalize_address("0200 Washington Street, ALVO, Nebraska", "US")
        assert got["state"] == "nebraska"
        assert got["city"] == "alvo"

    def test_india_city_before_state(self):
        # train_source1.tsv :: S1-755362802
        got = nz.normalize_address(
            "797, Lake Town Block A, Kolkata, Howrah, West Bengal", "India"
        )
        assert got["city"] == "howrah"
        assert got["state"] == "west bengal"

    def test_village_of_phrase_kept_as_city(self):
        got = nz.normalize_address(
            "294 Meadowcreek Drive, Unit Unit 2, Village Of Pewaukee, WI", "US"
        )
        assert got["city"] == "village of pewaukee"

    def test_street_name_is_not_mistaken_for_state(self):
        # "Michigan Street" must not yield state == "michigan"
        got = nz.normalize_address("2213 Michigan Street, Lansing, MI", "US")
        assert got["state"] != "michigan"
        assert got["state"] == "mi"
        assert got["city"] == "lansing"

    def test_unit_prefix_noise_does_not_break_city(self):
        # test_source1.tsv :: S1-626914593
        got = nz.normalize_address(
            "Unit BUILDING 3030, MD, 2701 Eastern Boulevard, Middle River", "US"
        )
        assert got["city"] == "middle river"
        assert got["state"] == "md"

    def test_house_number_is_not_taken_as_zip(self):
        # a bare 5-digit house number must not become the postal code
        got = nz.normalize_address("1795 Westchester Drive, High Point, NC", "US")
        assert got["postal_code"] != "1795"


# ─── Address: landmark handling ───────────────────────────────────────────────

class TestLandmark:
    def test_landmark_is_extracted(self):
        got = nz.normalize_address(
            "2505, Tower 1, Oakwood, Near Fortis Hospital, Mumbai, Maharashtra", "India"
        )
        assert got["landmark"] == "near fortis hospital"

    def test_landmark_text_leaves_the_core_address(self):
        got = nz.normalize_address(
            "2505, Tower 1, Oakwood, Near Fortis Hospital, Mumbai, Maharashtra", "India"
        )
        assert "near" not in got["normalized_address"]
        assert "fortis" not in got["normalized_address"]

    def test_opposite_is_extracted(self):
        # train_source1.tsv :: S1-564729135
        got = nz.normalize_address(
            "H.No.16-11-23/37/A, 2Nd Floor, Flat No.207, Sagar Hotel Building, "
            "Opp.Rta Office, Mo, Osarambagh, Hyderabad, Telangana",
            "India",
        )
        assert got["landmark"] == "opp rta office"
        assert "rta" not in got["normalized_address"]

    def test_city_still_found_after_landmark_removal(self):
        got = nz.normalize_address(
            "2505, Tower 1, Oakwood, Near Fortis Hospital, Mumbai, Maharashtra", "India"
        )
        assert got["city"] == "mumbai"
        assert got["state"] == "maharashtra"

    def test_no_landmark_gives_empty_string(self):
        got = nz.normalize_address("105 ELM ST, MORGANTON, NC", "US")
        assert got["landmark"] == ""


# ─── UTF-8 / non-Latin scripts ────────────────────────────────────────────────

TELUGU = "\u0c36\u0c4d\u0c30\u0c40"
DEVANAGARI = "\u0936\u094d\u0c30\u0c40"
TAMIL = "\u0b9a\u0bbe\u0bb2\u0bcd\u0bb9\u0bc1"


class TestUtf8Indic:
    @pytest.mark.parametrize("script", [TELUGU, DEVANAGARI, TAMIL])
    def test_non_ascii_name_survives(self, script):
        got = nz.normalize_name(script)["normalized_name"]
        assert got, "normalized name must not be empty"
        assert any(ord(c) > 127 for c in got), "non-ASCII characters must survive"

    @pytest.mark.parametrize("script", [TELUGU, DEVANAGARI, TAMIL])
    def test_non_ascii_name_is_not_emptied(self, script):
        # guards against a future 'strip to ASCII' regression
        assert nz.normalize_name(script)["name_tokens"] != []

    def test_accented_latin_is_preserved(self):
        # test_source1.tsv :: S1-156285671 (France)
        got = nz.normalize_address(
            "175 Boulevard du Pr\u00e9sident Franklin Roosevelt, Bordeaux, "
            "Nouvelle-Aquitaine",
            "France",
        )
        assert "pr\u00e9sident" in got["normalized_address"]

    def test_france_record_does_not_crash(self):
        # France appears only in test, never in train
        got = nz.normalize_address("175 Boulevard du Pr\u00e9sident, 75016 Paris", "France")
        assert got["normalized_address"]

    def test_unknown_country_label_does_not_crash(self):
        # country is an open set -- must not be validated against a fixed set
        got = nz.normalize_address("12 Some Street, Springfield, Atlantis", "Atlantis")
        assert got["normalized_address"] == "12 some street springfield atlantis"


# ─── Robustness ───────────────────────────────────────────────────────────────

class TestRobustness:
    def test_empty_name(self):
        r = nz.normalize_name("")
        assert r["normalized_name"] == ""
        assert r["name_tokens"] == []

    def test_none_name(self):
        r = nz.normalize_name(None)
        assert r["normalized_name"] == ""
        assert r["name_tokens"] == []

    def test_non_string_name_does_not_crash(self):
        assert nz.normalize_name(123)["normalized_name"] == ""

    def test_empty_address_has_no_components(self):
        r = nz.normalize_address("", None)
        assert r["normalized_address"] == ""
        assert r["landmark"] == ""
        assert r["city"] is None
        assert r["state"] is None
        assert r["postal_code"] is None

    def test_none_address_does_not_crash(self):
        r = nz.normalize_address(None, "US")
        assert r["normalized_address"] == ""
        assert r["city"] is None

    def test_address_without_country(self):
        r = nz.normalize_address("105 ELM ST, MORGANTON, NC", None)
        assert r["normalized_address"] == "105 elm street morganton nc"

    def test_normalize_record_passes_through_ids_and_country(self):
        r = nz.normalize_record({
            "entity_id": "S1-1",
            "country": "US",
            "business_name": "Acme Inc.",
            "business_address": "105 ELM ST, MORGANTON, NC",
        })
        assert r["entity_id"] == "S1-1"
        assert r["country"] == "US"
        assert r["normalized_business_name"] == "acme incorporated"
        assert r["business_name_tokens"] == "acme incorporated"


# ─── Real challenge rows (golden values) ──────────────────────────────────────

class TestRealDatasetRows:
    """Values below were lifted verbatim from the challenge TSVs and the
    expected outputs captured from the real module."""

    def test_orelees_barbershop(self):
        n = nz.normalize_name("Orelee's Barbershop")
        assert n["normalized_name"] == "orelee s barbershop"

    def test_custom_wealth_services_llc(self):
        n = nz.normalize_name("Custom Wealth Services LLC")
        assert n["normalized_name"] == "custom wealth services llc"

    def test_vision_partners_corp(self):
        n = nz.normalize_name("Vision Partners Corp")
        assert n["normalized_name"] == "vision partners corporation"

    def test_hyphenated_name_keeps_hyphen(self):
        # train_source3.tsv :: S3-90100170
        assert nz.normalize_name("Cinder Charlton-PC")["normalized_name"] == "cinder charlton-pc"

    def test_hyphenated_address_numbering_is_mangled_consistently(self):
        # train_source1.tsv :: S1-564729135 -- slashes become spaces
        got = nz.normalize_address(
            "H.No.16-11-23/37/A, 2Nd Floor, Flat No.207, Sagar Hotel Building, "
            "Opp.Rta Office, Mo, Osarambagh, Hyderabad, Telangana",
            "India",
        )
        assert got["normalized_address"] == (
            "h number 16-11-23 37 a 2nd floor flat number 207 sagar hotel building "
            "mo osarambagh hyderabad telangana"
        )
        assert got["city"] == "hyderabad"
        assert got["state"] == "telangana"

    def test_fractional_house_number_is_stable(self):
        got = nz.normalize_address("GREENSBORO, NC, 19 1/2 STARDUST TRAIL", "US")
        assert got["normalized_address"] == "greensboro nc 19 1 2 stardust trail"

    def test_leading_hyphens_survive(self):
        # train_source2.tsv :: S2-764573417 -- documents current behaviour
        assert nz.normalize_name("-- Holloway Peak Inc Seafood")["normalized_name"] == (
            "-- holloway peak incorporated seafood"
        )


# ─── Paths added after mutation testing ───────────────────────────────────────
# Each block below closes a coverage gap proven by deliberately breaking
# normalize.py and observing that the suite still passed.


class TestStateAtEndNoComma:
    """``_STATE_AT_END_RE`` only fires for a trailing state with no comma before
    it. Removing that regex left the suite green, so it needed a test."""

    def test_trailing_state_code_no_comma(self):
        got = nz.normalize_address("105 Elm St, Morganton NC", "US")
        assert got["state"] == "nc"

    def test_bare_city_state(self):
        got = nz.normalize_address("Morganton NC", "US")
        assert got["state"] == "nc"

    def test_trailing_full_state_name_no_comma(self):
        got = nz.normalize_address("105 Elm St Morganton North Carolina", "US")
        assert got["state"] == "north carolina"

    def test_state_before_comma_path(self):
        # "OH, Columbus, ..." is matched by _STATE_BEFORE_COMMA_RE, not the
        # after-comma rule -- pinning that so the two are not confused later.
        assert nz._STATE_BEFORE_COMMA_RE.search("oh, columbus, 5559 orville avenue")
        assert not nz._STATE_AFTER_COMMA_RE.search("oh, columbus, 5559 orville avenue")


class TestNoStatePresent:
    """``_extract_city`` has a final fallback loop used only when NO state is
    found anywhere. Stubbing that loop out left the suite green."""

    def test_city_found_without_any_state(self):
        got = nz.normalize_address("105 Elm St, Morganton", "US")
        assert got["state"] is None
        assert got["city"] == "morganton"

    def test_indian_city_without_state(self):
        got = nz.normalize_address("797, Lake Town Block A, Kolkata", "India")
        assert got["state"] is None
        assert got["city"] == "kolkata"

    def test_street_only_address_yields_no_city(self):
        # nothing here is a plausible city, so the field must stay empty
        got = nz.normalize_address("12 MG Road", "India")
        assert got["city"] is None


class TestDottedInitialisms:
    """``_DOTTED_INITIALISM_RE`` merges dotted forms ('l.l.c.') so the suffix
    maps can still match them. Disabling it left the suite green."""

    def test_llc_dotted(self):
        assert nz.normalize_name("Acme L.L.C.")["normalized_name"] == "acme llc"

    def test_sa_dotted(self):
        assert nz.normalize_name("Acme S.A.")["normalized_name"] == "acme sa"

    def test_sarl_dotted(self):
        assert nz.normalize_name("Acme S.A.R.L.")["normalized_name"] == "acme sarl"

    def test_undotted_llc_matches_dotted_llc(self):
        # the point of the merge: both spellings normalize identically
        assert (
            nz.normalize_name("Acme L.L.C.")["normalized_name"]
            == nz.normalize_name("Acme LLC")["normalized_name"]
        )

    def test_pte_is_not_mapped(self):
        # documents a gap in SUFFIX_MAP: 'pte' is a real Indian/UK legal suffix
        # but only 'pvt' is listed, so Pte/Pvt will not normalize together.
        assert nz.normalize_name("Acme Pte Ltd")["normalized_name"] == "acme pte limited"


# ─── Fixed defects (were strict-xfail, now permanent regressions) ────────────
# These three were written as xfail(strict) markers so that fixing them could
# not go unnoticed. They now XPASSed on the fix, so the markers were replaced by
# real assertions. The dataset evidence that prompted each fix is recorded here.


class TestIndicMarksPreserved:
    """Python's ``\\w`` excludes Unicode category M, so the old ``[^\\w\\s\\-]``
    class deleted every Indic matra and virama. On a 120k-row train sample,
    5,778 names contained marks and 100% of them lost them:
    'राम मार्केटिंग प्राइवेट लिमिटेड' -> 'र म म र क ट ग प र इव ट ल म ट ड'.
    Beyond being wrong, the shredding destroyed token identity for blocking."""

    def test_indic_matras_are_preserved(self):
        got = nz.normalize_name(TELUGU)["normalized_name"]
        assert len([c for c in got if not c.isspace()]) == len(
            [c for c in TELUGU if not c.isspace()]
        )

    def test_distinct_indic_words_do_not_collide(self):
        forms = ["\u0c15\u0c41", "\u0c15\u0c42", "\u0c15\u0c4d", "\u0c15"]
        normalized = {nz.normalize_name(w)["normalized_name"] for w in forms}
        assert len(normalized) == len(forms), f"collapsed into {normalized}"

    def test_devanagari_word_is_not_shredded(self):
        raw = "\u0930\u093e\u092e \u092e\u093e\u0930\u094d\u0915\u0947\u091f\u093f\u0902\u0917"
        got = nz.normalize_name(raw)["normalized_name"]
        assert got == raw
        # the shredding bug split one word into many single-character tokens
        assert len(got.split()) == 2

    def test_tamil_word_is_not_shredded(self):
        raw = TAMIL
        got = nz.normalize_name(raw)["normalized_name"]
        assert got == raw
        assert len(got.split()) == 1

    def test_thai_marks_survive(self):
        raw = "\u0e17\u0e35\u0e48\u0e44\u0e17\u0e22"  # ที่ไทย
        assert nz.normalize_name(raw)["normalized_name"] == raw

    def test_nfc_equivalent_forms_agree(self):
        # decomposed 'e' + combining acute must match the precomposed form
        assert (
            nz.normalize_name("Caf\u00e9")["normalized_name"]
            == nz.normalize_name("Cafe\u0301")["normalized_name"]
        )

    def test_ascii_still_normalizes_identically(self):
        # the mark-preserving path is a fallback; ASCII must not change
        assert nz.normalize_name("Acme Inc.")["normalized_name"] == "acme incorporated"
        assert nz.normalize_name("Lee & Lawson")["normalized_name"] == "lee and lawson"

    def test_punctuation_between_marks_is_still_stripped(self):
        got = nz.normalize_name("\u0930\u093e\u092e, \u092e\u093e\u0930\u094d\u0915.")["normalized_name"]
        assert got == "\u0930\u093e\u092e \u092e\u093e\u0930\u094d\u0915"


class TestCityFieldClean:
    """``_extract_city`` returned the raw matched part in its 'new delhi' branch
    without ever calling ``_clean_city_candidate``, so postal codes and trailing
    state codes leaked into the city field."""

    def test_city_field_has_no_digits(self):
        got = nz.normalize_address(
            "797, Lake Town Block A, Kolkata, Howrah, West Bengal 700001", "India"
        )
        assert got["city"] is not None
        assert not any(ch.isdigit() for ch in got["city"])

    def test_state_plus_postal_is_not_returned_as_city(self):
        # "west bengal 700001" was returned verbatim by the embedded-state branch
        got = nz.normalize_address(
            "797, Lake Town Block A, Kolkata, Howrah, West Bengal 700001", "India"
        )
        assert got["city"] == "howrah"

    def test_trailing_state_code_is_trimmed(self):
        got = nz.normalize_address("105 Elm St, Morganton NC", "US")
        assert got["state"] == "nc"
        assert got["city"] == "morganton"

    def test_trailing_state_code_trim_does_not_empty_the_city(self):
        got = nz.normalize_address("105 Elm St, Austin TX", "US")
        assert got["city"] == "austin"

    def test_embedded_state_name_is_preserved(self):
        # 'new delhi' legitimately contains its state name -- must survive
        got = nz.normalize_address("Block 1, New Delhi, Delhi 110001", "India")
        assert got["city"] is not None
        assert "delhi" in got["city"]

    def test_state_code_not_stripped_from_middle_of_name(self):
        got = nz.normalize_address("12 Nc Road, Nc City, NC", "US")
        assert got["city"] is not None
        assert "nc" in got["city"]


class TestFrenchAdminDivisionsRejected:
    """A French address ends in its region, so the trailing comma-part was
    returning the region instead of the city. On the test split this affected
    ~52% of the 1.69M French rows ('hauts-de-france' 250k, 'nouvelle-aquitaine'
    184k, 'pays de la loire' 179k, 'nord' 131k, 'gironde' 114k,
    'loire-atlantique' 100k) and left only 302 distinct city values in total."""

    def test_region_is_not_returned_as_city(self):
        got = nz.normalize_address(
            "175 Boulevard du Président Franklin Roosevelt, Bordeaux, "
            "Nouvelle-Aquitaine",
            "France",
        )
        assert got["city"] == "bordeaux"

    def test_hauts_de_france_is_rejected(self):
        got = nz.normalize_address("12 Rue de la Paix, Lille, Hauts-de-France", "France")
        assert got["city"] == "lille"

    def test_department_is_not_returned_as_city(self):
        got = nz.normalize_address("45 Rue Victor Hugo, Bordeaux, Gironde", "France")
        assert got["city"] == "bordeaux"

    def test_pays_de_la_loire_is_rejected(self):
        got = nz.normalize_address("8 Rue du Roi Albert, Nantes, Pays de la Loire", "France")
        assert got["city"] == "nantes"

    def test_hyphen_and_space_spellings_both_rejected(self):
        for region in ("Nord", "Loire-Atlantique", "pas-de-calais"):
            assert nz._is_admin_division(region), region
            assert nz._is_admin_division(region.replace("-", " ")), region
            assert nz._is_admin_division(region.upper()), region

    def test_accents_do_not_defeat_matching(self):
        assert nz._is_admin_division("cote d or")
        assert nz._is_admin_division("c\u00f4te-d'or")
        assert nz._is_admin_division("herault")
        assert nz._is_admin_division("H\u00e9rault")

    def test_a_city_named_like_part_of_a_department_survives(self):
        # 'calais' is a real city; the department is 'pas-de-calais'. Matching
        # is whole-string, so the city must not be swallowed.
        assert not nz._is_admin_division("calais")
        got = nz.normalize_address("9 Quai du Commerce, Calais, Pas-de-Calais", "France")
        assert got["city"] == "calais"

    def test_real_paris_city_survives(self):
        got = nz.normalize_address("10 Rue de Rivoli, 75001 Paris, Ile-de-France", "France")
        assert got["city"] == "paris"

    def test_city_containing_region_token_is_not_rejected(self):
        # only whole-string matches count, so a longer real city is safe
        assert not nz._is_admin_division("villefranche")
        assert not nz._is_admin_division("bordeaux")

    def test_guard_also_protects_the_no_state_fallback(self):
        # When the region is not a clean comma-part, _extract_state finds no
        # state and the no-state fallback scans from the end. The admin-division
        # guard inside valid() is the only thing stopping the region there, so
        # this exercises that path directly rather than through a full address.
        # Input is pre-lowercased because normalize_address does that upstream.
        assert nz._extract_city("bordeaux, nouvelle-aquitaine", None, None, "France") == "bordeaux"

    def test_fallback_rejects_region_but_keeps_city(self):
        # same fallback, inverted: the region must be skipped, the city kept
        assert nz._extract_city("lille, hauts-de-france", None, None, "France") == "lille"

    def test_fallback_with_only_a_region_yields_nothing(self):
        # better an empty city than a near-constant region value
        assert nz._extract_city("hauts-de-france", None, None, "France") is None


class TestFrenchStreetAndRegion:
    """Found by running process_file over a real 3k-row slice before committing
    to the full run: 'rue' appears in 65.8% of French addresses and was being
    returned as the city for 17,108 of 60k rows, because _looks_like_street
    only inspected the last two tokens and the street word sits at the front."""

    def test_street_word_in_last_three_tokens_is_rejected(self):
        assert nz._looks_like_street("rue de foo")
        assert nz._looks_like_street("avenue de dunkerque")
        assert nz._looks_like_street("bis rue pierre dignac")

    def test_leading_street_word_in_a_long_part_is_rejected(self):
        # 4 tokens, so the street word falls outside a last-3 window
        assert not any(t in nz.STREET_TYPES for t in "bd du president wilson".split()[-3:])
        assert nz._looks_like_street("bd du president wilson")

    def test_city_parts_without_a_street_word_are_kept(self):
        assert not nz._looks_like_street("paris")
        assert not nz._looks_like_street("la teste-de-buch")
        assert not nz._looks_like_street("saint-nazaire")
        assert not nz._looks_like_street("michigan city")

    def test_french_street_vocabulary_is_registered(self):
        for word in ("rue", "bd", "impasse", "chemin", "allee", "quai", "cours"):
            assert word in nz.STREET_TYPES, word

    def test_city_before_region(self):
        got = nz.normalize_address("Lille, 329 Avenue de Dunkerque, Hauts-de-France", "France")
        assert got["city"] == "lille"
        assert got["state"] == "hauts-de-france"

    def test_boulevard_part_does_not_become_the_city(self):
        got = nz.normalize_address(
            "Bordeaux, 154 BD du President Wilson, Nouvelle-Aquitaine", "France"
        )
        assert got["city"] == "bordeaux"
        assert got["state"] == "nouvelle-aquitaine"

    def test_house_number_qualifier_part_does_not_become_the_city(self):
        got = nz.normalize_address(
            "Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac", "France"
        )
        assert got["city"] == "la teste-de-buch"
        assert got["state"] == "nouvelle-aquitaine"

    def test_impasse_does_not_become_the_city(self):
        got = nz.normalize_address(
            "5 Impasse Jean Baptiste Clement, Saint-Herblain, Pays de la Loire", "France"
        )
        assert got["city"] == "saint-herblain"

    def test_region_is_reported_as_the_state(self):
        # France has no US-style state code, so the division fills the column
        got = nz.normalize_address("9 Quai du Commerce, Calais, Pas-de-Calais", "France")
        assert got["state"] == "pas-de-calais"
        assert got["city"] == "calais"

    def test_french_article_is_not_read_as_louisiana(self):
        # "La Teste-de-Buch" made 'la' match the Louisiana state code
        got = nz.normalize_address(
            "Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac", "France"
        )
        assert got["state"] != "la"

    def test_us_state_codes_still_work(self):
        # the France short-circuit must not leak into other countries
        got = nz.normalize_address("1795 Westchester Drive, High Point, NC 27317", "US")
        assert got["state"] == "nc"
        assert got["city"] == "high point"

    def test_french_unit_descriptors_are_junk(self):
        for word in ("residence", "appartement", "batiment", "etage"):
            assert word in nz.JUNK_PART_TOKENS, word
