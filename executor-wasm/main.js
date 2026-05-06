/*
 * 版權所有 (c) 2019 年至今 Dmitry Stepanov 和 Fyrox Engine 貢獻者。
 *
 * 特此向任何獲得副本的人免費授予許可
 * 本軟體及相關文件文件（「軟體」）的，以處理
 * 不受限制地使用軟體，包括但不限於權利
 * 使用、複製、修改、合併、發布、分發、再授權和/或出售
 * 該軟體的副本，並允許該軟體的使用者
 * 可以這樣做，但須滿足以下條件：
 *
 * 上述版權聲明和本許可聲明應包含在所有內容中
 * 軟體的副本或主要部分。
 *
 * 軟體以「現況」提供，不提供任何形式的明示或保證
 * 默示的保證，包括但不限於適銷性保證，
 * 適用於特定目的且不侵權。在任何情況下都不得
 * 作者或版權所有者對任何索賠、損害或其他責任負責
 * 責任，無論是合約訴訟、侵權訴訟或其他訴訟，均由以下原因引起：
 * 與本軟體無關或與之相關，或與本軟體相關的使用或其他交易
 * SOFTWARE.
 */

const moduleGame = import('./pkg/executor_wasm.js').then(({ default: init, main }) =>
  init().then(() => main)
)
const elementTargetButton = document.querySelector('#button-start')
const elementMain = document.querySelector('#main')

const run = async () => {
  elementTargetButton.removeEventListener('click', run)
  elementMain.remove()

  const context = new AudioContext()

  if (context.state !== 'running') {
    await context.resume()
  }

  return (await moduleGame)()
}

elementTargetButton.addEventListener('click', run, {
  once: true,
  passive: true,
})
