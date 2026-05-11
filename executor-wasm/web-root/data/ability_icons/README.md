# Ability Icons

這個資料夾放新增 / 尚未有正式圖的技能圖示。Lua template 的 `icon` 欄位會指到實際 PNG，前端 HUD 會直接讀該路徑。

企劃換圖時請保持檔名不變，直接替換對應 PNG 即可。建議尺寸 `96x96` 或以上正方形 PNG，前端會縮放到 HUD 使用的 `64x64`。

目前預設檔名：

- `ability_default_placeholder.png`：缺圖 fallback。
- `ability_flame_blade.png`：炎刃。
- `ability_fire_dash.png`：火焰衝擊。
- `ability_flame_assault.png`：炎襲。
- `ability_matchlock_gun.png`：火繩銃。

雜賀孫市四個技能沿用原本已存在的圖：

- `omfx/data/hero1_1.png`：狙擊模式。
- `omfx/data/hero1_2.png`：雜賀援軍。
- `omfx/data/hero1_3.png`：雨鐵砲。
- `omfx/data/hero1_4.png`：三段擊。
