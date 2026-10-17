"""Generate the FRAME / CLIP / VIDEO training set from paraphrase templates.

Each template is a question pattern with slots. Filling the slots with many different people,
objects, actions and events teaches the classifier to key on *what kind of evidence* a question
asks for rather than on content words. `template_id` is kept so validation can hold out whole
templates (a much harder and more honest test than a random split).

Labeling rubric (applied by construction):
  FRAME   static visual fact at one moment: colour, clothing, on-screen text, object identity,
          spatial layout, how many things are visible at once
  CLIP    dynamic fact in a short segment: action, event, cause/effect, before/after,
          what someone says or does at a point, when something happens
  VIDEO   holistic fact about the whole video: topic, purpose, genre, message, tone,
          relationships, conclusions, generic "which statement is true" stems
  FRAME+CLIP   static attribute of something picked out by an event ("colour of the car that crashes")
  CLIP+VIDEO   counting / ordering / progression of events across the video
  FRAME+VIDEO  static things aggregated across the video ("how many different cars appear")
  Descriptors that only identify someone ("the man in the red shirt") add no label.

Run:  python src/templates.py   ->   src/data/train.csv
"""
import csv
import random
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "data" / "train.csv"
EXCLUDE = [ROOT / "src" / "data" / "heldout.csv", ROOT / "src" / "data" / "seeds.csv"]

# ----------------------------------------------------------------------------- slot values
PEOPLE = ["the man", "the woman", "the boy", "the girl", "the host", "the presenter", "the speaker",
          "the chef", "the player", "the referee", "the driver", "the dancer", "the singer",
          "the teacher", "the student", "the reporter", "the athlete", "the doctor", "the scientist",
          "the main character", "the interviewer", "the guest", "the coach", "the old man",
          "the young woman", "the vlogger", "the police officer", "the shopkeeper", "the goalkeeper",
          "the pianist", "the magician", "the mechanic", "the child", "the farmer", "the nurse",
          "the pilot", "the waiter", "the climber", "the artist", "the soldier"]
ANIMALS = ["the dog", "the cat", "the bird", "the horse", "the monkey", "the puppy", "the bear", "the fox"]
OBJECTS = ["the car", "the box", "the cup", "the bag", "the bottle", "the phone", "the laptop",
           "the book", "the ball", "the chair", "the table", "the poster", "the bicycle", "the camera",
           "the guitar", "the umbrella", "the knife", "the flag", "the trophy", "the lamp", "the clock",
           "the door", "the bowl", "the plate", "the vase", "the helmet", "the backpack", "the truck",
           "the boat", "the cake", "the map", "the key", "the mug", "the painting", "the robot",
           "the drone", "the microphone", "the sofa", "the tent", "the bus", "the kite", "the suitcase"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "gray",
          "brown", "silver"]
CLOTHING = ["shirt", "jacket", "hat", "dress", "shoes", "scarf", "coat", "tie", "uniform", "gloves",
            "sweater", "cap", "skirt", "apron", "hoodie"]
CLOTHING_PL = ["shirts", "jackets", "hats", "dresses", "outfits", "costumes", "uniforms", "scarves"]
PLACES = ["the kitchen", "the stage", "the street", "the office", "the classroom", "the field",
          "the beach", "the park", "the studio", "the garage", "the restaurant", "the store", "the lab",
          "the court", "the arena", "the forest", "the hospital", "the airport", "the living room",
          "the market", "the gym", "the library", "the station"]
SURFACES = ["the sign", "the poster", "the screen", "the whiteboard", "the banner", "the slide",
            "the label", "the T-shirt", "the scoreboard", "the menu", "the book cover", "the box",
            "the jersey", "the truck", "the billboard", "the title card", "the newspaper",
            "the license plate", "the door", "the cup"]
THINGS = ["people", "cars", "chairs", "players", "dancers", "birds", "flags", "trees", "buildings",
          "cups", "candles", "balloons", "lamps", "screens", "windows", "boats", "students", "musicians"]
EVENT_COUNTS = ["goals", "jumps", "laps", "songs", "experiments", "interviews", "scene changes",
                "penalties", "dishes", "tricks", "fouls", "questions", "rounds", "attempts",
                "costume changes", "location changes", "explosions", "speeches"]
VTYPES = ["video", "film", "documentary", "lecture", "tutorial", "vlog", "interview", "talk",
          "presentation", "match", "performance", "show", "episode", "movie", "news report",
          "story", "lesson", "program", "broadcast"]
TOPICS = ["climate change", "healthy eating", "teamwork", "online privacy", "city life",
          "ancient history", "space exploration", "friendship", "the economy", "education",
          "renewable energy", "the history of the city", "the artist's career", "the team's season",
          "artificial intelligence", "family", "tourism", "traditional culture"]
PAIRS = ["the two teams", "the two methods", "the two characters", "the two cities",
         "the two products", "the first half and the second half", "the old and new designs",
         "the two approaches", "past and present", "the two speakers"]
PROCESSES = ["the experiment", "the recipe", "the project", "the journey", "the renovation",
             "the competition", "the training program", "the construction", "the repair",
             "the cooking process", "the race", "the investigation"]
BEHAVIORS = ["running speed", "technique", "movements", "driving", "dance moves", "fighting style",
             "playing style", "pace", "posture", "strategy on the field"]
EVENTS = ["the music stops", "the lights go out", "the whistle blows", "the alarm rings",
          "the guest arrives", "the race begins", "the crowd cheers", "the glass breaks",
          "the rain starts", "the car crashes", "the speech ends", "the bell rings",
          "the door opens", "the phone rings", "the game starts", "the curtain rises",
          "the ball hits the window", "the dog barks", "the train arrives", "the fireworks start",
          "the teacher enters", "the timer goes off", "the power goes out", "the song ends",
          "the referee blows the whistle", "the bus pulls away", "the water boils",
          "the balloon pops", "the camera zooms in", "the host introduces the guest"]
EVENT_NOUNS = ["the speech", "the race", "the interview", "the experiment", "the performance",
               "the match", "the storm", "the dance", "the demonstration", "the meeting",
               "the ceremony", "the concert", "the cooking segment", "the fight scene",
               "the presentation", "the penalty kick", "the first round", "the car chase",
               "the workout", "the lecture", "the second goal", "the halftime break"]
ONGOING = ["the music plays", "the crowd cheers", "the presenter speaks", "the rain falls",
           "the car drives past", "the guests eat", "the band performs",
           "the teacher explains the problem", "the water boils", "the video plays"]
ACTIONS = ["opens the door", "picks up the box", "enters the room", "sits down", "starts running",
           "jumps over the fence", "throws the ball", "falls to the ground", "turns around",
           "waves at the camera", "leaves the house", "starts the engine", "pours the water",
           "cuts the vegetables", "scores a goal", "crosses the finish line", "lifts the weight",
           "hugs the child", "presses the button", "raises a hand", "takes off the hat",
           "puts on the jacket", "climbs the ladder", "drops the cup", "opens the box",
           "kicks the ball", "points at the screen", "writes on the board", "plays the guitar",
           "answers the phone", "shakes hands with the guest", "walks onto the stage",
           "gets into the car", "bows to the audience", "fixes the bike", "lights the candle"]
ACTIONS_BASE = ["open the door", "pick up the box", "enter the room", "sit down", "jump",
                "throw the ball", "fall", "turn around", "wave", "leave the house",
                "start the engine", "pour the water", "cut the vegetables", "score a goal",
                "cross the finish line", "lift the weight", "press the button", "raise a hand",
                "take off the hat", "climb the ladder", "drop the cup", "kick the ball",
                "point at the screen", "write on the board", "play the guitar", "answer the phone",
                "clap", "laugh", "smile", "nod", "run", "dance", "sing", "shoot the ball",
                "pass the ball", "change the slide", "stir the pot", "check the phone", "cry",
                "shout"]
OBJECT_EVENTS = ["fall", "break", "move", "stop", "explode", "roll away", "catch fire", "tip over",
                 "light up", "start spinning", "sink", "shake"]
CONTAINERS = ["bag", "box", "drawer", "fridge", "pocket", "backpack", "car", "oven", "suitcase"]
SOUNDS = ["the bell", "the alarm", "the horn", "the whistle", "a dog barking", "the phone ringing",
          "applause", "the siren", "laughter", "the theme song"]
FINAL_EVENTS = ["the final goal", "the team's victory", "the ending", "the explosion at the end",
                "the argument at the end", "the final scene", "the character's decision",
                "the winner being announced"]
SPORT_ACTIONS = ["scores the first goal", "scores the winning goal", "commits the foul",
                 "takes the penalty", "wins the race", "makes the final shot", "gets the red card",
                 "catches the pass"]
ROUND_NOUNS = ["round", "lap", "inning", "set", "game", "song", "act", "stage"]
# (noun, relative clause) pairs: "What color is the {noun} that {clause}?"
REL = [("box", "the man picks up"), ("ball", "the player kicks into the goal"), ("cup", "the girl drops"),
       ("car", "crashes into the wall"), ("bag", "the woman carries out"), ("book", "the teacher holds up"),
       ("bottle", "the boy throws"), ("phone", "rings on the table"), ("door", "the guest walks through"),
       ("hat", "the magician pulls the rabbit from"), ("vase", "falls off the shelf"),
       ("cake", "the chef cuts"), ("kite", "the girl flies"), ("letter", "the man reads aloud"),
       ("truck", "drives past the camera"), ("balloon", "pops"), ("plate", "the waiter drops"),
       ("umbrella", "the woman opens"), ("jacket", "the singer takes off"), ("ladder", "the worker climbs"),
       ("trophy", "the captain lifts"), ("dog", "chases the ball"), ("bird", "flies out of the cage"),
       ("boat", "sinks"), ("flag", "the athlete waves"), ("bike", "the boy rides away on")]

SLOTS = {
    "person": PEOPLE, "person2": PEOPLE, "object": OBJECTS, "color": COLORS, "clothing": CLOTHING,
    "clothing_pl": CLOTHING_PL, "place": PLACES, "surface": SURFACES, "things": THINGS,
    "event_counts": EVENT_COUNTS, "vtype": VTYPES, "topic": TOPICS, "pair": PAIRS,
    "process": PROCESSES, "behavior": BEHAVIORS, "event": EVENTS, "event2": EVENTS,
    "event_noun": EVENT_NOUNS, "ongoing": ONGOING, "action": ACTIONS, "action_base": ACTIONS_BASE,
    "object_event": OBJECT_EVENTS, "container": CONTAINERS, "sound": SOUNDS,
    "final_event": FINAL_EVENTS, "sport_action": SPORT_ACTIONS, "round_noun": ROUND_NOUNS,
}

# ----------------------------------------------------------------------------- templates
FRAME = [
    "What color is {object}?", "What color is {person}'s {clothing}?",
    "What color {clothing} is {person} wearing?", "What is {person} wearing?",
    "What is {person} wearing on their head?", "What is written on {surface}?",
    "What text appears on {surface}?", "What word is shown on {surface}?",
    "What number is displayed on {surface}?", "What does {surface} say?", "What brand is {object}?",
    "Which logo is printed on {object}?", "What is the name shown on {surface}?",
    "What is placed on top of {object}?", "What object is next to {object}?",
    "What is to the left of {object}?", "What is on the right side of {person}?",
    "Where is {object} located in the room?", "What is behind {person}?",
    "What can be seen in the background behind {person}?", "Which animal can be seen in {place}?",
    "What kind of {object_noun} is shown?", "What type of {object_noun} is {person} holding?",
    "What is {person} holding?", "What is {person} holding in their hand?",
    "How many {things} are visible in this scene?", "How many {things} can be seen in the frame?",
    "How many {things} are in {place} at that moment?", "What is the facial expression of {person}?",
    "What shape is {object}?", "What material is {object} made of?", "Is {person} wearing glasses?",
    "What does {object} look like?", "Which hand is {person} using to hold {object}?",
    "What is displayed on the screen behind {person}?", "What time is shown on the clock?",
    "Which instrument is {person} playing?", "What is the color of {person}'s hair?",
    "In which room is {person} standing?", "What is printed on {person}'s {clothing}?",
    "What is the score shown on the scoreboard?", "Which country's flag is displayed?",
    "What type of building appears in the background?", "What is the weather like in the scene?",
    "What is the name of the shop on the sign?", "What food is on the plate in front of {person}?",
    "What is the person in the {color} {clothing} holding?", "Which of the following colors is {object}?",
    "Which of the following objects is on the table?", "What color is {object} in the first scene?",
    "What is written on {surface} at the end of the video?",
    "What is {person} wearing at the beginning of the video?", "What is shown in the opening shot?",
    "What is displayed on the screen in the last scene?", "What logo appears in the final frame?",
    "What is the overall shape of {object}?", "What are all the players wearing?",
    "What is the whole team wearing in the group photo?", "What is {person} wearing at {ts}?",
    "What is shown on {surface} at {ts}?", "What is the color of the {object_noun} on the left?",
    "What does the label on {object} say?", "Where is {person} sitting?",
    "What animal is sitting on {object}?",
]
CLIP = [
    "What does {agent} do after {event}?", "What does {agent} do before {event}?",
    "What does {agent} do right after {event}?", "What happens after {agent} {action}?",
    "What happens right before {event}?", "What happens when {event}?",
    "How does {agent} react when {event}?", "How does {person} respond when {event}?",
    "Why does {agent} {action_base}?", "Why did {agent} {action_base}?",
    "What causes {agent} to {action_base}?", "What is {agent} doing while {ongoing}?",
    "How does {agent} {action_base}?", "What action does {agent} perform after {event}?",
    "What does {agent} do with {object}?", "What does {agent} do next?",
    "When does {agent} {action_base}?", "At what moment does {agent} {action_base}?",
    "What does {person} say after {event}?", "What does {person} say about {object}?",
    "What does {person} mention right before {event}?", "What sound is heard when {event}?",
    "What happens to {object} after {event}?", "What happens to {object}?",
    "Where does {agent} go after {event}?", "Who {action} after {event}?",
    "What is {agent} trying to do?", "What is the purpose of {person}'s action after {event}?",
    "What step comes after {agent} {action}?", "In what way does {agent} {action_base}?",
    "What does {agent} do immediately after {event}?",
    "What is the result of {person}'s attempt to {action_base}?",
    "How does {person} respond to {person2}'s question?", "What does {person} ask {person2}?",
    "What does {agent} do at the end of the video?", "What happens at the beginning of the video?",
    "How does the video end?", "What happens in the first minute of the video?",
    "What happens between {ts} and {ts2}?", "What does {agent} do at {ts}?",
    "What is {person} doing in {place}?", "Why does {object} {object_event}?",
    "What makes {object} {object_event}?", "What does {agent} do when {person2} {action}?",
    "Which of the following does {agent} do after {event}?",
    "What does {agent} do first when entering {place}?",
    "What is the last thing {agent} does before {event}?", "How does {person} {action_base} in {place}?",
    "What does {person} explain while {ongoing}?", "How does {agent} move when {event}?",
    "What happens during {event_noun}?", "What does {person} do during {event_noun}?",
    "Why does {person} stop during {event_noun}?", "What is the reaction of the crowd when {event}?",
    "Which direction does {agent} move after {event}?",
]
VIDEO = [
    "What is the main topic of the {vtype}?", "What is the {vtype} mainly about?",
    "What is the main idea of the {vtype}?", "What is the purpose of this {vtype}?",
    "What genre does this {vtype} belong to?", "What is the overall message of the {vtype}?",
    "Which of the following best summarizes the {vtype}?", "What is the central theme of the {vtype}?",
    "What lesson does the {vtype} teach?", "What conclusion does {person} reach by the end?",
    "How does the story develop over the course of the {vtype}?",
    "How does {person}'s attitude change throughout the {vtype}?",
    "What is the relationship between {person} and {person2}?",
    "How are the different parts of the {vtype} connected?", "How is the {vtype} structured?",
    "Which of the following statements about the {vtype} is true?",
    "Which of the following is not mentioned in the {vtype}?",
    "Which statement is correct according to the {vtype}?",
    "What can be inferred about {person} from the {vtype}?",
    "What is the {vtype} trying to persuade viewers to do?", "Who is the intended audience of the {vtype}?",
    "What is the main argument made in the {vtype}?",
    "How does the mood of the {vtype} shift from beginning to end?",
    "What is the overall tone of the {vtype}?", "What problem does the {vtype} discuss?",
    "What does the {vtype} suggest about {topic}?", "How does {person} feel about {topic} overall?",
    "What is the {vtype}'s view on {topic}?", "Which of the following topics is discussed in the {vtype}?",
    "What is the best title for this {vtype}?", "What type of {vtype} is this?",
    "Where does the {vtype} mostly take place?", "How does the {vtype} compare {pair}?",
    "What is the main difference between {pair} presented in the {vtype}?",
    "What do the scenes in the {vtype} have in common?", "What is the narrator's main point?",
    "What does {person} conclude at the end of the talk?", "Summarize what the {vtype} is about.",
    "What is the overarching story of the {vtype}?", "How does the plot unfold?",
    "What is the moral of the story?", "What is the significance of {object} in the {vtype}?",
    "What is {person}'s attitude toward {topic}?", "In what order are the topics presented in the {vtype}?",
    "What is the final outcome of {process}?", "How does {person} change as a character over the {vtype}?",
    "Which of the following is true?", "What can be concluded from the {vtype}?",
    "What is the {vtype}'s main message about {topic}?", "Why did the creator make this {vtype}?",
    "What is {person}'s main goal throughout the {vtype}?",
]
FRAME_CLIP = [
    "What color is the {rel_noun} that {rel_clause}?", "What is written on the {rel_noun} that {rel_clause}?",
    "What does the {rel_noun} that {rel_clause} look like?", "What brand is the {rel_noun} that {rel_clause}?",
    "What is {person} wearing when {event}?", "What color is {object} when {event}?",
    "What is {person} holding when {event}?", "What is displayed on the screen when {event}?",
    "What number is on the jersey of the player who {sport_action}?",
    "What is the expression on {person}'s face after {event}?",
    "Which object does {agent} pick up after {event}?", "What item does {person} take out of the {container}?",
    "What is written on the sign that {person} walks past?",
    "What color is the {clothing} that {person} puts on?", "Where does {person} put {object} after {event}?",
    "What appears on the screen right after {event}?", "What is the score on the scoreboard after {event_noun}?",
    "What is the person who {action} wearing?", "What does {agent} pick up?",
    "What does {person} grab before {event}?", "Which {object_noun} does {person} choose?",
]
CLIP_VIDEO = [
    "How many times does {agent} {action_base}?", "How many times does {agent} {action_base} in the video?",
    "How many {event_counts} are there in the {vtype}?", "How many {event_counts} happen throughout the {vtype}?",
    "What is the total number of {event_counts} in the {vtype}?", "In what order do the events happen?",
    "Which happens first: {event} or {event2}?", "What is the sequence of {person}'s actions in the {vtype}?",
    "How does {person}'s {behavior} change from the first attempt to the last?",
    "What does {agent} do repeatedly throughout the {vtype}?",
    "Which action is repeated most often in the {vtype}?", "How often does {person} {action_base}?",
    "What events lead up to {final_event}?", "How does {process} progress from start to finish?",
    "What are the main stages of {process}?", "How does {person}'s {behavior} change over the {vtype}?",
    "What happens in the first and last scenes of the {vtype}?",
    "Which of the following events happens last in the {vtype}?",
    "What does {person} do each time {event}?", "How many times is {sound} heard?",
    "When does {agent} {action_base} for the second time?", "What steps does {person} follow in the {vtype}?",
    "How do {person}'s actions in the first half compare with the second half?",
]
FRAME_VIDEO = [
    "How many different {things} appear in the {vtype}?", "Which color appears most often in the {vtype}?",
    "What objects appear in both the first and last scenes?", "Which logos are shown throughout the {vtype}?",
    "How does {person}'s appearance change throughout the {vtype}?",
    "What {clothing_pl} does {person} wear over the course of the {vtype}?",
    "Which of the following objects appears in the {vtype}?",
    "Which of the following items is not shown in the {vtype}?", "How many people appear in the {vtype}?",
    "What is the most common {object_noun} color in the {vtype}?", "Which locations are shown in the {vtype}?",
    "What text appears repeatedly on the screen?",
    "How does the setting change visually from the first scene to the last?",
    "Which characters appear in every scene?", "What is the dominant color scheme of the {vtype}?",
    "Does {object} appear more than once in the {vtype}?",
    "How many {things} are shown in total throughout the {vtype}?", "What animals can be seen in the {vtype}?",
    "Which of the following brands is shown in the {vtype}?",
]
FRAME_CLIP_VIDEO = [
    "What is {person} wearing each time {event}?", "What color is the car involved in each crash?",
    "Which player in the {color} jersey scores the most goals?", "What does the scoreboard show after each {round_noun}?",
    "How does {person}'s outfit look after each {round_noun}?", "What is written on the screen every time {event}?",
]
GROUPS = [  # (templates, (frame, clip, video), samples per template)
    (FRAME, (1, 0, 0), 8), (CLIP, (0, 1, 0), 8), (VIDEO, (0, 0, 1), 8), (FRAME_CLIP, (1, 1, 0), 6),
    (CLIP_VIDEO, (0, 1, 1), 6), (FRAME_VIDEO, (1, 0, 1), 6), (FRAME_CLIP_VIDEO, (1, 1, 1), 6),
]
PREFIXES = ["In the video, ", "According to the video, ", "Based on the video, ", "In this video, ",
            "From the video, "]


def fill(template, rng):
    """Fill every slot in a template with a random value (distinct values for paired slots)."""
    values = {}
    for _, field, _, _ in string.Formatter().parse(template):
        if not field or field in values:
            continue
        if field == "agent":
            values[field] = rng.choice(ANIMALS) if rng.random() < 0.15 else rng.choice(PEOPLE)
        elif field == "object_noun":
            values[field] = rng.choice(OBJECTS)[4:]
        elif field in ("rel_noun", "rel_clause"):
            noun, clause = rng.choice(REL)
            values.update(rel_noun=noun, rel_clause=clause)
        elif field in ("ts", "ts2"):
            minute, second = rng.randint(0, 9), rng.randint(0, 59)
            values["ts"] = f"{minute}:{second:02d}"
            values["ts2"] = f"{minute + rng.randint(0, 2)}:{min(59, second + rng.randint(5, 30)):02d}"
        else:
            values[field] = rng.choice(SLOTS[field])
    for a, b in (("person", "person2"), ("event", "event2")):
        while a in values and b in values and values[a] == values[b]:
            values[b] = rng.choice(SLOTS[b])
    text = template.format(**values)
    return text[0].upper() + text[1:]


def augment(question, rng):
    """Scope-neutral variation so phrases like 'in the video' do not become class cues."""
    lowered = question.lower()
    if rng.random() < 0.2:
        return rng.choice(PREFIXES) + question[0].lower() + question[1:]
    neutral = not any(w in lowered for w in ("video", "how many", "scene", "shot", "frame", "time"))
    if neutral and question.endswith("?") and rng.random() < 0.12:
        return question[:-1] + " in the video?"
    return question


def build(seed=7):
    rng = random.Random(seed)
    excluded = set()
    for path in EXCLUDE:
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                excluded |= {row["question"].strip().lower() for row in csv.DictReader(f)}
    rows, labels_seen = {}, {}
    for group_index, (templates, labels, per_template) in enumerate(GROUPS):
        for t_index, template in enumerate(templates):
            template_id = f"g{group_index}t{t_index}"
            for _ in range(per_template * 3):  # oversample, keep the first distinct fills
                if sum(1 for r in rows.values() if r[4] == template_id) >= per_template:
                    break
                question = augment(fill(template, rng), rng)
                key = question.lower()
                if key in excluded:
                    continue
                labels_seen.setdefault(key, set()).add(labels)
                rows.setdefault(key, (question, *labels, template_id))
    # Drop any question generated under two different label sets (ambiguous by construction).
    clean = [row for key, row in rows.items() if len(labels_seen[key]) == 1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "frame", "clip", "video", "template_id"])
        writer.writerows(sorted(clean, key=lambda r: (r[4], r[0])))
    return clean


if __name__ == "__main__":
    data = build()
    multi = sum(1 for r in data if sum(r[1:4]) > 1)
    print(f"Wrote {len(data)} questions from {len({r[4] for r in data})} templates "
          f"({multi / len(data):.0%} multi-label) to {OUT.relative_to(ROOT)}")
